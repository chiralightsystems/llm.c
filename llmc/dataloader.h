/*
Implements:
- DataLoader for model training. Reads and serves data shards.
- EvalLoader for multiple-choice evaluation datasets, e.g. HellaSwag.
*/
#ifndef DATALOADER_H
#define DATALOADER_H

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <ctype.h>
#include <errno.h>
#include <assert.h>
#include <string.h>
#include <sys/types.h>
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck
// defines: mallocCheck
#include "utils.h"
#include "rand.h"

// ----------------------------------------------------------------------------
// implementation of glob for Windows is in dev/unistd.h
#ifndef _WIN32
#include <glob.h>
#endif
// ----------------------------------------------------------------------------
// Distributed Data Loader
#define HEADER_SIZE 256

typedef enum {
    DATALOADER_TOKEN_FORMAT_LLMC_UINT16 = 0,
    DATALOADER_TOKEN_FORMAT_NUMPY_UINT32 = 1,
} DataLoaderTokenFormat;

static const char* dataloader_token_format_name(DataLoaderTokenFormat format) {
    switch (format) {
        case DATALOADER_TOKEN_FORMAT_LLMC_UINT16: return "llmc_uint16";
        case DATALOADER_TOKEN_FORMAT_NUMPY_UINT32: return "numpy_uint32";
        default: return "unknown";
    }
}

typedef struct {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    int process_rank;
    int num_processes;
    // batch and token information
    size_t B;
    size_t T;
    size_t num_tokens; // total number of tokens
    size_t shard_num_samples;  // total number of samples in the current shard per process
    // shards and current position
    glob_t glob_result; // stores the result of glob, for all shards we want to iterate
    size_t current_shard_idx; // the current shard we are reading from
    size_t current_sample_idx; // the current sample we are reading from
    // file handle
    FILE* tokens_file;
    // data buffers
    uint16_t* buffer; // we fread data from file into this buffer
    uint32_t* buffer_u32; // direct NumPy uint32 input_ids path
    int* inputs;  // input tokens into transformer
    int* targets; // target tokens for the transformer
    // random shuffle related variables
    mt19937_state shuffle_rng;
    int should_shuffle;
    // Canonical fixed-row traversal. This is deliberately separate from the
    // legacy unshuffled flat-stream path: it reads complete T-token rows,
    // wraps by row without dropping a tail batch, and derives row-local
    // next-token targets whose final (masked) target is a valid placeholder.
    int row_aligned_sequential;
    size_t logical_row_count;
    size_t numpy_rows;
    size_t numpy_columns;
    // Fast identity guard for exact resume. This fingerprints the ordered
    // source paths plus their validated layout/extent; the repo launcher also
    // binds the manifest-declared payload SHA-256 for content identity.
    uint64_t source_fingerprint;
    int* shard_indices;
    int* intra_shard_indices;
    // sizes in bytes
    size_t total_batch_size_bytes;  // total across all processes
    size_t local_batch_offset_bytes;  // inner-sample offset for this process
    size_t header_bytes;  // header size in bytes
    size_t token_size_bytes;
    int64_t file_size_bytes;
    DataLoaderTokenFormat token_format;
} DataLoader;

static void dataloader_seek_(FILE* file, int64_t offset, int whence) {
#ifdef _WIN32
    int status = _fseeki64(file, offset, whence);
#else
    int status = fseeko(file, (off_t)offset, whence);
#endif
    if (status != 0) {
        fprintf(stderr, "Error: failed to seek to byte offset %lld in token data\n",
                (long long)offset);
        exit(EXIT_FAILURE);
    }
}

static int64_t dataloader_tell_(FILE* file) {
#ifdef _WIN32
    __int64 offset = _ftelli64(file);
#else
    off_t offset = ftello(file);
#endif
    if (offset < 0) {
        fprintf(stderr, "Error: failed to query token data file position\n");
        exit(EXIT_FAILURE);
    }
    return (int64_t)offset;
}

static uint32_t dataloader_u32_le_(const unsigned char* bytes) {
    return ((uint32_t)bytes[0]) |
           ((uint32_t)bytes[1] << 8) |
           ((uint32_t)bytes[2] << 16) |
           ((uint32_t)bytes[3] << 24);
}

static void dataloader_fingerprint_bytes_(
        uint64_t* fingerprint,
        const void* data,
        size_t byte_count) {
    const unsigned char* bytes = (const unsigned char*)data;
    for (size_t index = 0; index < byte_count; ++index) {
        *fingerprint ^= (uint64_t)bytes[index];
        *fingerprint *= UINT64_C(1099511628211);
    }
}

static void dataloader_fingerprint_u64_(uint64_t* fingerprint, uint64_t value) {
    unsigned char bytes[8];
    for (int index = 0; index < 8; ++index) {
        bytes[index] = (unsigned char)((value >> (8 * index)) & UINT64_C(0xff));
    }
    dataloader_fingerprint_bytes_(fingerprint, bytes, sizeof(bytes));
}

static const char* dataloader_numpy_dict_value_(const char* header, const char* key) {
    char single_quoted[64];
    char double_quoted[64];
    snprintf(single_quoted, sizeof(single_quoted), "'%s'", key);
    snprintf(double_quoted, sizeof(double_quoted), "\"%s\"", key);
    const char* entry = strstr(header, single_quoted);
    if (entry == NULL) {
        entry = strstr(header, double_quoted);
    }
    if (entry == NULL) {
        return NULL;
    }
    const char* value = strchr(entry, ':');
    if (value == NULL) {
        return NULL;
    }
    ++value;
    while (isspace((unsigned char)*value)) {
        ++value;
    }
    return value;
}

static int dataloader_numpy_shape_2d_(
        const char* header,
        uint64_t* rows,
        uint64_t* columns) {
    const char* cursor = dataloader_numpy_dict_value_(header, "shape");
    if (cursor == NULL || *cursor != '(') {
        return 0;
    }
    ++cursor;
    while (isspace((unsigned char)*cursor)) {
        ++cursor;
    }
    errno = 0;
    char* end = NULL;
    unsigned long long parsed_rows = strtoull(cursor, &end, 10);
    if (end == cursor || errno == ERANGE || parsed_rows == 0) {
        return 0;
    }
    cursor = end;
    while (isspace((unsigned char)*cursor)) {
        ++cursor;
    }
    if (*cursor++ != ',') {
        return 0;
    }
    while (isspace((unsigned char)*cursor)) {
        ++cursor;
    }
    errno = 0;
    unsigned long long parsed_columns = strtoull(cursor, &end, 10);
    if (end == cursor || errno == ERANGE || parsed_columns == 0) {
        return 0;
    }
    cursor = end;
    while (isspace((unsigned char)*cursor)) {
        ++cursor;
    }
    if (*cursor == ',') {
        ++cursor;
        while (isspace((unsigned char)*cursor)) {
            ++cursor;
        }
    }
    if (*cursor != ')') {
        return 0;
    }
    *rows = (uint64_t)parsed_rows;
    *columns = (uint64_t)parsed_columns;
    return 1;
}

static int64_t dataloader_load_numpy_uint32_(
        DataLoader* loader,
        const unsigned char prefix[12],
        const char* filename) {
    int major = (int)prefix[6];
    size_t preamble_bytes;
    uint32_t header_length;
    int minor = (int)prefix[7];
    if (major == 1 && minor == 0) {
        preamble_bytes = 10;
        header_length = (uint32_t)prefix[8] | ((uint32_t)prefix[9] << 8);
    } else if (major == 2 && minor == 0) {
        preamble_bytes = 12;
        header_length = dataloader_u32_le_(prefix + 8);
    } else {
        fprintf(stderr, "Error: unsupported NumPy format version %d.%d in %s\n",
                major, minor, filename);
        exit(EXIT_FAILURE);
    }
    if (header_length == 0 || header_length > 1024U * 1024U) {
        fprintf(stderr, "Error: invalid NumPy header length %u in %s\n",
                header_length, filename);
        exit(EXIT_FAILURE);
    }

    char* header = (char*)mallocCheck((size_t)header_length + 1);
    dataloader_seek_(loader->tokens_file, (int64_t)preamble_bytes, SEEK_SET);
    freadCheck(header, 1, header_length, loader->tokens_file);
    header[header_length] = '\0';
    const char* dtype = dataloader_numpy_dict_value_(header, "descr");
    int has_uint32_le = dtype != NULL &&
        ((*dtype == '\'' && strncmp(dtype, "'<u4'", 5) == 0) ||
         (*dtype == '"' && strncmp(dtype, "\"<u4\"", 5) == 0));
    const char* fortran_order = dataloader_numpy_dict_value_(header, "fortran_order");
    int has_c_order = fortran_order != NULL && strncmp(fortran_order, "False", 5) == 0;
    uint64_t rows = 0;
    uint64_t columns = 0;
    int has_2d_shape = dataloader_numpy_shape_2d_(header, &rows, &columns);
    if (!has_uint32_le || !has_c_order || !has_2d_shape) {
        fprintf(stderr,
                "Error: NumPy token data must be a C-contiguous 2-D little-endian uint32 array: %s\n",
                filename);
        free(header);
        exit(EXIT_FAILURE);
    }
    free(header);

    loader->header_bytes = preamble_bytes + (size_t)header_length;
    loader->token_size_bytes = sizeof(uint32_t);
    loader->token_format = DATALOADER_TOKEN_FORMAT_NUMPY_UINT32;
    loader->numpy_rows = (size_t)rows;
    loader->numpy_columns = (size_t)columns;
    dataloader_seek_(loader->tokens_file, 0, SEEK_END);
    loader->file_size_bytes = dataloader_tell_(loader->tokens_file);
    if (rows > UINT64_MAX / columns) {
        fprintf(stderr, "Error: NumPy token shape overflows: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    uint64_t token_count = rows * columns;
    if (token_count > ((uint64_t)INT64_MAX - loader->header_bytes) / sizeof(uint32_t)) {
        fprintf(stderr, "Error: NumPy token payload is too large: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    int64_t expected_file_size =
        (int64_t)loader->header_bytes + (int64_t)(token_count * sizeof(uint32_t));
    if (loader->file_size_bytes != expected_file_size) {
        fprintf(stderr, "Error: NumPy token shape does not match file size: %s\n", filename);
        exit(EXIT_FAILURE);
    }
    return (int64_t)token_count;
}

int64_t dataloader_load_shard_(DataLoader *loader, int shard_index) {
    if (loader->should_shuffle) {
        shard_index = loader->shard_indices[shard_index];
    }
    // use the first glob match as the filename for now
    const char* filename = loader->glob_result.gl_pathv[shard_index];
    // open the input file for reading. also only a single file can be opened at a time
    if (loader->tokens_file != NULL) {
        fcloseCheck(loader->tokens_file);
    }
    loader->tokens_file = fopenCheck(filename, "rb");
    unsigned char prefix[12];
    freadCheck(prefix, 1, sizeof(prefix), loader->tokens_file);
    dataloader_seek_(loader->tokens_file, 0, SEEK_SET);
    int64_t ntok;
    const unsigned char numpy_magic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
    if (memcmp(prefix, numpy_magic, sizeof(numpy_magic)) == 0) {
        ntok = dataloader_load_numpy_uint32_(loader, prefix, filename);
    } else {
        // validate the legacy llm.c uint16 shard header
        int header[HEADER_SIZE];
        freadCheck(header, sizeof(int), HEADER_SIZE, loader->tokens_file);
        if (header[0] != 20240520) {
            printf("Bad magic in the data file\n");
            printf("---> HINT: expected an llm.c uint16 shard or NumPy uint32 input_ids array.\n");
            exit(EXIT_FAILURE);
        }
        if (header[1] != 1) { printf("Bad version in data file\n"); exit(EXIT_FAILURE); }
        ntok = header[2];
        loader->header_bytes = HEADER_SIZE * sizeof(int);
        loader->token_size_bytes = sizeof(uint16_t);
        loader->token_format = DATALOADER_TOKEN_FORMAT_LLMC_UINT16;
        dataloader_seek_(loader->tokens_file, 0, SEEK_END);
        loader->file_size_bytes = dataloader_tell_(loader->tokens_file);
        int64_t expected_file_size =
            (int64_t)loader->header_bytes + ntok * (int64_t)sizeof(uint16_t);
        if (loader->file_size_bytes != expected_file_size) {
            printf("Error: file size is not as expected\n");
            exit(EXIT_FAILURE);
        }
    }
    assert(ntok > 0); // we expect some tokens in the file. this should never trip, right?
    loader->total_batch_size_bytes =
        (loader->num_processes * (loader->B * loader->T)) * loader->token_size_bytes;
    loader->local_batch_offset_bytes =
        loader->process_rank * loader->B * loader->T * loader->token_size_bytes;
    // -1 token due to taking B*T+1 tokens while moving by B*T tokens.
    loader->shard_num_samples =
        (size_t)(ntok - 1) / (loader->num_processes * loader->B * loader->T);
    return ntok;
}

void prepare_intra_shard_indices_(DataLoader *loader) {
    // shuffle the examples inside the shards
    if (loader->intra_shard_indices != NULL) {
        // in case shards have different number of samples / sizes
        free(loader->intra_shard_indices);
    }
    loader->intra_shard_indices = (int*)mallocCheck(loader->shard_num_samples * sizeof(int));
    init_identity_permutation(loader->intra_shard_indices, (int) loader->shard_num_samples);
    random_permutation(loader->intra_shard_indices, (int) loader->shard_num_samples, &loader->shuffle_rng);
}

void dataloader_reset(DataLoader *loader) {
    loader->current_shard_idx = 0;
    loader->current_sample_idx = 0;

    if (loader->should_shuffle) {  // shuffle the shards
        random_permutation(loader->shard_indices, (int) loader->glob_result.gl_pathc, &loader->shuffle_rng);
    }

    dataloader_load_shard_(loader, (int) loader->current_shard_idx);

    if (loader->should_shuffle) {
        prepare_intra_shard_indices_(loader);
    }
}

void dataloader_advance_(DataLoader *loader) {
    if (loader->current_shard_idx == loader->glob_result.gl_pathc - 1) {
        // if we are at the last shard, we reset the loader and start a new epoch
        dataloader_reset(loader);
        return;
    }

    // advance the loader by loading the next data shard and resetting the position
    loader->current_shard_idx = (loader->current_shard_idx + 1) % loader->glob_result.gl_pathc;
    loader->current_sample_idx = 0;
    dataloader_load_shard_(loader, (int) loader->current_shard_idx);

    if (loader->should_shuffle) {
        prepare_intra_shard_indices_(loader);
    }
}

void dataloader_init_with_policy(DataLoader *loader,
                                 const char* filename_pattern,
                                 size_t B,
                                 size_t T,
                                 int process_rank,
                                 int num_processes,
                                 int should_shuffle,
                                 int row_aligned_sequential,
                                 int row_aligned_eval_regroup) {
    if (B == 0 || T == 0 || num_processes <= 0 ||
        process_rank < 0 || process_rank >= num_processes) {
        fprintf(stderr, "Error: invalid dataloader B, T, rank, or process count\n");
        exit(EXIT_FAILURE);
    }
    if (should_shuffle && row_aligned_sequential) {
        fprintf(stderr, "Error: row-aligned sequential loading cannot also shuffle\n");
        exit(EXIT_FAILURE);
    }
    if (row_aligned_eval_regroup && !row_aligned_sequential) {
        fprintf(stderr, "Error: row-aligned eval regrouping requires row-aligned sequential loading\n");
        exit(EXIT_FAILURE);
    }
    loader->process_rank = process_rank;
    loader->num_processes = num_processes;
    loader->B = B;
    loader->T = T;
    loader->tokens_file = NULL;
    loader->should_shuffle = should_shuffle;
    loader->row_aligned_sequential = row_aligned_sequential;
    loader->logical_row_count = 0;
    loader->numpy_rows = 0;
    loader->numpy_columns = 0;
    loader->source_fingerprint = UINT64_C(14695981039346656037);
    loader->shard_indices = NULL;
    loader->intra_shard_indices = NULL;

    // glob to get the list of files matching the pattern, these are our data shards
    int glob_status = glob(filename_pattern, 0, NULL, &loader->glob_result);
    if (glob_status != 0) {
        printf("Error: failed to glob pattern: %s\n", filename_pattern);
        exit(EXIT_FAILURE);
    }
    if (loader->glob_result.gl_pathc == 0) {
        printf("Error: no files found matching the pattern: %s\n", filename_pattern);
        exit(EXIT_FAILURE);
    }
    if (row_aligned_sequential && loader->glob_result.gl_pathc != 1) {
        fprintf(stderr,
                "Error: canonical row-aligned sequential loading currently requires exactly one direct NumPy cache file; matched %zu files\n",
                loader->glob_result.gl_pathc);
        exit(EXIT_FAILURE);
    }

    if (should_shuffle) {
        mt19937_state shuffle_rng;
        manual_seed(&shuffle_rng, 42 + process_rank);
        loader->shuffle_rng = shuffle_rng;
        loader->shard_indices = (int*)mallocCheck(loader->glob_result.gl_pathc * sizeof(int));
        init_identity_permutation(loader->shard_indices, (int) loader->glob_result.gl_pathc);
        loader->intra_shard_indices = NULL;  // dynamically allocated allowing different shard sizes
    }

    // inspect and validate all shards so we don't get any runtime errors later
    // if too slow / too many shards, may wish to revisit later
    int64_t ntok_total = 0;
    for (int shard_index = 0; shard_index < loader->glob_result.gl_pathc; shard_index++) {
        const char* source_path = loader->glob_result.gl_pathv[shard_index];
        dataloader_fingerprint_bytes_(
            &loader->source_fingerprint,
            source_path,
            strlen(source_path) + 1);
        int64_t shard_ntok = dataloader_load_shard_(loader, shard_index);
        dataloader_fingerprint_u64_(
            &loader->source_fingerprint, (uint64_t)loader->file_size_bytes);
        dataloader_fingerprint_u64_(
            &loader->source_fingerprint, (uint64_t)loader->header_bytes);
        dataloader_fingerprint_u64_(
            &loader->source_fingerprint, (uint64_t)shard_ntok);
        dataloader_fingerprint_u64_(
            &loader->source_fingerprint, (uint64_t)loader->token_format);
        if (row_aligned_sequential) {
            if (loader->token_format != DATALOADER_TOKEN_FORMAT_NUMPY_UINT32) {
                fprintf(stderr,
                        "Error: canonical row-aligned sequential loading requires a direct NumPy uint32 cache\n");
                exit(EXIT_FAILURE);
            }
            const int logical_rows_fit_inside_physical_rows =
                loader->numpy_columns % T == 0;
            const int logical_rows_group_complete_physical_rows =
                row_aligned_eval_regroup && T % loader->numpy_columns == 0;
            const size_t trailing_tokens = (size_t)shard_ntok % T;
            if ((!logical_rows_fit_inside_physical_rows &&
                 !logical_rows_group_complete_physical_rows) ||
                (trailing_tokens != 0 && !row_aligned_eval_regroup)) {
                fprintf(stderr,
                        "Error: row_reset sequential loading requires the NumPy row width (%zu) to tile T (%zu); only eval may regroup complete physical rows and drop a shorter tail\n",
                        loader->numpy_columns, T);
                exit(EXIT_FAILURE);
            }
            loader->logical_row_count = (size_t)shard_ntok / T;
            if (row_aligned_eval_regroup && loader->process_rank == 0 &&
                (!logical_rows_fit_inside_physical_rows || trailing_tokens != 0)) {
                printf(
                    "Row-aligned eval regroup: physical row width %zu -> logical T %zu; ignoring %zu trailing token(s).\n",
                    loader->numpy_columns,
                    T,
                    trailing_tokens);
            }
            if (B > SIZE_MAX / (size_t)num_processes) {
                fprintf(stderr, "Error: row-aligned global batch row count overflows size_t\n");
                exit(EXIT_FAILURE);
            }
            const size_t global_batch_rows = B * (size_t)num_processes;
            if (loader->logical_row_count < global_batch_rows) {
                fprintf(
                    stderr,
                    "Error: row-aligned sequential cache has %zu logical rows but the distributed batch requires at least %zu disjoint rows\n",
                    loader->logical_row_count,
                    global_batch_rows);
                exit(EXIT_FAILURE);
            }
        } else {
            // The legacy flat/shuffled path reads B*T+1 tokens per batch.
            assert(shard_ntok >= (int64_t) (num_processes * B * T + 1));
        }
        ntok_total += shard_ntok;
    }
    // debugging prints
    // printf("DataLoader: filename_pattern: %s\n", filename_pattern);
    // printf("DataLoader: Found %ld tokens across %zu shards\n", ntok_total, loader->glob_result.gl_pathc);

    // allocate all the space we'll need
    loader->buffer = (uint16_t*)mallocCheck((B * T + 1) * sizeof(uint16_t));
    loader->buffer_u32 = (uint32_t*)mallocCheck((B * T + 1) * sizeof(uint32_t));
    loader->inputs = (int*)mallocCheck(B * T * sizeof(int));
    loader->targets = (int*)mallocCheck(B * T * sizeof(int));
    loader->num_tokens = ntok_total;

    // reset the loader, to initialize it
    dataloader_reset(loader);
}

void dataloader_init(DataLoader *loader,
                     const char* filename_pattern,
                     size_t B,
                     size_t T,
                     int process_rank,
                     int num_processes,
                     int should_shuffle) {
    dataloader_init_with_policy(
        loader,
        filename_pattern,
        B,
        T,
        process_rank,
        num_processes,
        should_shuffle,
        0,
        0);
}

void dataloader_load_row_aligned_batch_(DataLoader* loader) {
    assert(loader->row_aligned_sequential);
    assert(loader->glob_result.gl_pathc == 1);
    assert(loader->token_format == DATALOADER_TOKEN_FORMAT_NUMPY_UINT32);
    assert(loader->logical_row_count > 0);

    const size_t B = loader->B;
    const size_t T = loader->T;
    const size_t token_count = loader->logical_row_count * T;
    const size_t rank_row_offset = (size_t)loader->process_rank * B;
    size_t token_index =
        ((loader->current_sample_idx + rank_row_offset) % loader->logical_row_count) * T;
    size_t remaining = B * T;
    size_t output_offset = 0;
    while (remaining > 0) {
        const size_t contiguous = remaining < token_count - token_index
            ? remaining
            : token_count - token_index;
        const int64_t byte_offset = (int64_t)loader->header_bytes +
            (int64_t)(token_index * sizeof(uint32_t));
        dataloader_seek_(loader->tokens_file, byte_offset, SEEK_SET);
        freadCheck(
            loader->buffer_u32 + output_offset,
            sizeof(uint32_t),
            contiguous,
            loader->tokens_file);
        output_offset += contiguous;
        remaining -= contiguous;
        token_index = 0;
    }

    for (size_t row = 0; row < B; row++) {
        for (size_t column = 0; column < T; column++) {
            const size_t index = row * T + column;
            const uint32_t input_token = loader->buffer_u32[index];
            if (input_token > INT32_MAX) {
                fprintf(stderr, "Error: uint32 token id exceeds the signed int runtime range\n");
                exit(EXIT_FAILURE);
            }
            loader->inputs[index] = (int)input_token;
            // The trainer masks column T-1 under row_reset. Keep a valid,
            // row-local placeholder there so token validation never observes
            // a cross-row target.
            const size_t target_index = column + 1 < T ? index + 1 : index;
            const uint32_t target_token = loader->buffer_u32[target_index];
            if (target_token > INT32_MAX) {
                fprintf(stderr, "Error: uint32 token id exceeds the signed int runtime range\n");
                exit(EXIT_FAILURE);
            }
            loader->targets[index] = (int)target_token;
        }
    }
}

void dataloader_load_batch(DataLoader* loader) {
    assert(!loader->should_shuffle || (loader->should_shuffle && loader->intra_shard_indices != NULL));
    assert(loader->current_sample_idx < loader->shard_num_samples);
    size_t idx = loader->should_shuffle ? loader->intra_shard_indices[loader->current_sample_idx] : loader->current_sample_idx;
    size_t global_batch_offset_bytes = idx * loader->total_batch_size_bytes;
    int64_t current_offset = loader->header_bytes + global_batch_offset_bytes + loader->local_batch_offset_bytes;

    size_t B = loader->B;
    size_t T = loader->T;
    dataloader_seek_(loader->tokens_file, current_offset, SEEK_SET);
    if (loader->token_format == DATALOADER_TOKEN_FORMAT_NUMPY_UINT32) {
        freadCheck(loader->buffer_u32, sizeof(uint32_t), B*T+1, loader->tokens_file);
        for (size_t i = 0; i < B*T; i++) {
            if (loader->buffer_u32[i] > INT32_MAX || loader->buffer_u32[i + 1] > INT32_MAX) {
                fprintf(stderr, "Error: uint32 token id exceeds the signed int runtime range\n");
                exit(EXIT_FAILURE);
            }
            loader->inputs[i] = (int)loader->buffer_u32[i];
            loader->targets[i] = (int)loader->buffer_u32[i+1];
        }
    } else {
        freadCheck(loader->buffer, sizeof(uint16_t), B*T+1, loader->tokens_file);
        for (size_t i = 0; i < B*T; i++) {
            loader->inputs[i] = (int)loader->buffer[i];
            loader->targets[i] = (int)loader->buffer[i+1];
        }
    }
}

void dataloader_next_batch(DataLoader *loader) {
    if (loader->row_aligned_sequential) {
        dataloader_load_row_aligned_batch_(loader);
        const size_t global_rows = (size_t)loader->num_processes * loader->B;
        loader->current_sample_idx =
            (loader->current_sample_idx + global_rows) % loader->logical_row_count;
        return;
    }
    // if the next batch would go past the end of the file, advance the loader
    if (loader->current_sample_idx >= loader->shard_num_samples) {
        dataloader_advance_(loader);
    }
    dataloader_load_batch(loader);
    loader->current_sample_idx += 1;
}


void dataloader_resume(DataLoader *loader, size_t current_shard_idx, size_t current_sample_idx) {
    // used during model resumption (-y 1) flag
    if (loader->row_aligned_sequential) {
        if (current_shard_idx != 0 || current_sample_idx >= loader->logical_row_count) {
            fprintf(stderr, "Error: invalid row-aligned sequential dataloader cursor\n");
            exit(EXIT_FAILURE);
        }
        loader->current_shard_idx = 0;
        loader->current_sample_idx = current_sample_idx;
        dataloader_load_shard_(loader, 0);
        return;
    }
    loader->current_shard_idx = current_shard_idx;
    loader->current_sample_idx = current_sample_idx;
    dataloader_load_shard_(loader, (int) loader->current_shard_idx);
}

void dataloader_free(DataLoader *loader) {
    free(loader->buffer);
    free(loader->buffer_u32);
    free(loader->inputs);
    free(loader->targets);
    if (loader->should_shuffle) {
        free(loader->shard_indices);
        free(loader->intra_shard_indices);
    }
    fcloseCheck(loader->tokens_file);
    globfree(&loader->glob_result);
}

// ----------------------------------------------------------------------------
// Distributed Eval Loader
// Many evals (like) HellaSwag and MMLU are multiple-choice
// where there are 4 possible continuations and a label for the correct one
// We want to load and serve these style of evals
/*
Copy pasting the section on the eval datafile format, from data_common.py:
- First comes a header with 256 int32s
- The examples follow, each example is a stream of uint16_t:
    - <START_EXAMPLE> delimiter of 2**16-1, i.e. 65,535
    - <EXAMPLE_BYTES>, bytes encoding this example, allowing efficient skip to next
    - <EXAMPLE_INDEX>, the index of the example in the dataset
    - <LABEL>, the index of the correct completion
    - <NUM_COMPLETIONS>, indicating the number of completions (usually 4)
    - <NUM><CONTEXT_TOKENS>, where <NUM> is the number of tokens in the context
    - <NUM><COMPLETION_TOKENS>, repeated NUM_COMPLETIONS times
*/

// for now, could relax later
#define ASSUMED_NUM_COMPLETIONS 4
// helper macro for ceildiv
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

typedef struct {
    // variables related to distributed training
    // each process/worker has to access different parts of the data
    int process_rank;
    int num_processes;
    // hyperparameters. use size_t to prevent overflow
    size_t B; // (micro) batch size dimension of the tensor that feeds into the model
    size_t T; // maximum context length of the model
    // input handling and its state
    FILE* eval_file;
    uint16_t* buffer; // we fread data from file into this buffer
    // public variables that could be accessed from outside
    int num_examples; // in total across all processes
    int num_batches; // to process the entire dataset across all processes
    int start_example_index; // the assignment of work for this process, start
    int end_example_index; // and end. start is inclusive, end is exclusive
    int current_example_index; // the next example we would read
    int* inputs;  // input tokens into transformer
    int* targets; // target tokens for the transformer
    char* mask; // mask=1 at all completion token locations
    int* label; // the correct completion labels
    int num_completions; // number of completions for this example
} EvalLoader;

void evalloader_reset(EvalLoader *loader) {
    // we have to be careful that each process starts at the correct offset.
    // For example if there are N examples in the file and 4 processes,
    // then process 0 should start at 0, process 1 at N/4, process 2 at N/2, etc.
    // determine how much work there is for all processes
    int examples_per_process = CEIL_DIV(loader->num_examples, loader->num_processes);
    int can_fit_examples = (int) (loader->B / ASSUMED_NUM_COMPLETIONS);
    if (can_fit_examples == 0) {
        // this could be fixed in the future, but for now keeping it simple and throw error when B too low
        printf("HellaSwag EvalLoader: batch size %zu is < %d\n", loader->B, ASSUMED_NUM_COMPLETIONS);
        printf("---> HINT: Disable HellaSwag eval with -h 0, or increase batch size with -b\n");
        exit(EXIT_FAILURE);
    }
    loader->num_batches = CEIL_DIV(examples_per_process, can_fit_examples);
    // determine the start and end example indices for this process
    loader->start_example_index = examples_per_process * loader->process_rank;
    loader->end_example_index = examples_per_process * (loader->process_rank + 1);
    // crop the end example index to the total number of examples
    if (loader->end_example_index > loader->num_examples) {
        loader->end_example_index = loader->num_examples;
    }
    // now seek through the file to the start of that example
    // utilize <EXAMPLE_BYTES> for efficiency
    int64_t header_bytes = HEADER_SIZE * sizeof(int);
    fseekCheck(loader->eval_file, (int) header_bytes, SEEK_SET);
    for (int i = 0; i < loader->start_example_index; i++) {
        uint16_t example_header[3];
        // read 3 uint16_t values: <START_EXAMPLE>, <EXAMPLE_BYTES>, <EXAMPLE_INDEX>
        freadCheck(&example_header[0], sizeof(uint16_t), 3, loader->eval_file);
        // validate the <START_EXAMPLE> delimiter
        assert(example_header[0] == 65535); // <START_EXAMPLE> delimiter
        // validate the <EXAMPLE_INDEX>
        assert(example_header[2] == i); // <EXAMPLE_INDEX> should match the loop index
        // skip to the next example, keeping in mind that we already read the header
        size_t remaining_bytes = example_header[1] - sizeof(uint16_t) * 3;
        assert(remaining_bytes > 0); // we expect some bytes in the example
        fseekCheck(loader->eval_file, (int) remaining_bytes, SEEK_CUR);
    }
    // now we are at the start of the example we want to start at, pointing at <START_EXAMPLE>
    loader->current_example_index = loader->start_example_index;
}

void evalloader_init(EvalLoader *loader,
                     const char* filename,
                     size_t B,
                     size_t T,
                     int process_rank,
                     int num_processes) {
    loader->process_rank = process_rank;
    loader->num_processes = num_processes;
    loader->B = B;
    loader->T = T;

    // open the file and validate the header
    loader->eval_file = fopenCheck(filename, "rb");
    // validate the header
    int header[HEADER_SIZE];
    freadCheck(header, sizeof(int), HEADER_SIZE, loader->eval_file);
    if (header[0] != 20240522) { printf("Bad magic in eval file\n"); exit(EXIT_FAILURE); }
    if (header[1] != 1) { printf("Bad version in data file\n"); exit(EXIT_FAILURE); }
    loader->num_examples = header[2]; // number of examples in the file
    assert(loader->num_examples >= num_processes); // avoid headaches for now
    size_t longest_example_bytes = header[3]; // longest example in the file
    // basic sensibility check we could relax later. but roughly each example
    // contains the prompt (or "context") and 4 completions, all of these have to be
    // up to T tokens, and their tokens are uint16_t (so 2 bytes/token).
    // There's a few more things in each example but they are minor.
    // So longest example should be roughly this. Just trying to make sure it's sensible.
    assert(longest_example_bytes > 0 && longest_example_bytes < (1+ASSUMED_NUM_COMPLETIONS)*T*2);

    // allocate all the space we'll need
    int can_fit_examples = (int) (B / ASSUMED_NUM_COMPLETIONS);
    loader->buffer = (uint16_t*)mallocCheck(longest_example_bytes);
    loader->inputs = (int*)calloc(B * T, sizeof(int));
    loader->targets = (int*)calloc(B * T, sizeof(int));
    loader->mask = (char*)mallocCheck(B * T * sizeof(char));
    loader->label = (int*)mallocCheck(can_fit_examples * sizeof(int));

    // reset the loader, to initialize it
    evalloader_reset(loader);
}

void evalloader_next_example_(EvalLoader *loader, int example_batch_index) {
    // this function populates the inputs, targets, mask, and label fields for one example
    // because every (B,T) tensor can fit multiple examples and we want to take advantage,
    // we also pass in the example_batch_index to indicate which example in the batch we are loading
    // and each example takes up ASSUMED_NUM_COMPLETIONS rows in the batch
    size_t B = loader->B;
    size_t T = loader->T;
    int batch_dim_offset = example_batch_index * ASSUMED_NUM_COMPLETIONS;
    // read the current example header
    uint16_t example_header[3];
    freadCheck(&example_header[0], sizeof(uint16_t), 3, loader->eval_file);
    // validate the <START_EXAMPLE> delimiter
    assert(example_header[0] == 65535); // <START_EXAMPLE> delimiter
    // validate the <EXAMPLE_INDEX>
    assert(example_header[2] == loader->current_example_index); // <EXAMPLE_INDEX> should match the loop index
    assert(example_header[2] >= loader->start_example_index && example_header[2] < loader->end_example_index);
    // read the rest of the example (we have space for 3 more uint16_t values in buffer, it's ok)
    size_t example_bytes = example_header[1] - sizeof(uint16_t) * 3;
    // read example_bytes into buffer. careful that this is actually in the units of bytes
    freadCheck(loader->buffer, sizeof(char), example_bytes, loader->eval_file);
    // process the example label
    int label = (int)loader->buffer[0];
    int can_fit_examples = (int) (loader->B / ASSUMED_NUM_COMPLETIONS);
    assert(label >= 0 && label < ASSUMED_NUM_COMPLETIONS); // we expect the label to be in [0, 4) for right now
    assert(example_batch_index >= 0 && example_batch_index < can_fit_examples);
    loader->label[example_batch_index] = label; // store for output
    // process the number of completions
    int num_completions = (int)loader->buffer[1];
    assert(num_completions == ASSUMED_NUM_COMPLETIONS); // we expect 4 completions for now
    assert(batch_dim_offset + num_completions <= B); // we expect to fit in the batch
    loader->num_completions = num_completions; // store for output
    // process the context
    // the context is shared for all completions, so we insert it into all data rows equally
    int context_length = (int)loader->buffer[2];
    uint16_t *context_tokens_start = &loader->buffer[3]; // where the tokens start
    assert(context_length > 0 && context_length < T); // context is non-empty and up to T
    for (int b = 0; b < num_completions; b++) {
        for (int i = 0; i < context_length; i++) {
            int boff = batch_dim_offset + b;
            int tok_cur = (int)context_tokens_start[i];
            loader->inputs[boff * T + i] = tok_cur;
        }
    }
    // process the completions, insert them in their row, right after the (shared) context
    uint16_t *completions_iter = loader->buffer + 3 + context_length;
    for (int c = 0; c < num_completions; c++) {
        int coff = batch_dim_offset + c;
        int completion_length = (int)completions_iter[0];
        uint16_t *completion_tokens_start = completions_iter + 1;
        assert(completion_length > 0 && context_length + completion_length < T); // things fit?
        for (int i = 0; i < completion_length; i++) {
            int tok_cur = (int)completion_tokens_start[i];
            // at inputs, the completions simply follow the context
            loader->inputs[coff * T + context_length + i] = tok_cur;
            // at targets things start to get tricky
            // we expect the last context token to predict the first completion token
            // and then onwards from there.
            loader->targets[coff * T + context_length + i - 1] = tok_cur;
            // and at these positions, we want to set mask=1, because these are the
            // positions where we want to average the loss, in each row, to determine
            // its overall probability of following the context.
            loader->mask[coff * T + context_length + i - 1] = 1;
        }
        completions_iter += 1 + completion_length; // move to the next completion
    }
    // advance the current example to point to the next one we'd load
    loader->current_example_index += 1;
}

void evalloader_next_batch(EvalLoader *loader) {
    size_t B = loader->B;
    size_t T = loader->T;
    // init mask to zeros, no need to do it for inputs & targets, the values where the mask
    // is set will be correctly overwritten every time.
    memset(loader->mask, 0, B * T * sizeof(char));
    // ok here is the problem we are solving
    // we have a batch dimension of B, which we want to take full advantage of
    // each example has some number of completions (usually 4)
    // so we want to pack as many examples into rows of B as we can fit
    int can_fit_examples = (int) (B / ASSUMED_NUM_COMPLETIONS); // how many examples can we fit in the batch?
    for (int i = 0; i < can_fit_examples; i++) {
        if (loader->current_example_index >= loader->end_example_index) {
            break; // this process has exhausted its work, noop from here on
        }
        evalloader_next_example_(loader, i);
    }
}

int evalloader_stat_losses(EvalLoader *loader, float* losses) {
    // compute statistics of losses (B*T) resulting from a forward pass
    // on a batch that was constructed from EvalLoader
    // putting this functionality here because it is tightly coupled
    // with how we construct and represent the data batches.
    // returns the number of correct examples in this batch.
    int correct = 0;
    size_t B = loader->B;
    size_t T = loader->T;
    // iterate the examples in this batch
    int can_fit_examples = (int) (B / ASSUMED_NUM_COMPLETIONS);
    for (int i = 0; i < can_fit_examples; i++) {
        float min_loss = 0.0f;
        int min_loss_index = -1;
        char active = 0; // is this example active or fully empty?
        // iterate the completions in this example
        for (int b = 0; b < ASSUMED_NUM_COMPLETIONS; b++) {
            int boff = i * ASSUMED_NUM_COMPLETIONS + b;
            // evaluate the quality of this completion
            // its quality is simply the average loss over the tokens
            float average_loss = 0.0f;
            int count = 0;
            for (int t = 0; t < T; t++) {
                char mask = loader->mask[boff * T + t];
                if (mask == 1) {
                    active = 1;
                    average_loss += losses[boff * T + t];
                    count++;
                }
            }
            if (count > 0) { average_loss /= count; }
            if (b == 0 || average_loss < min_loss) {
                min_loss = average_loss;
                min_loss_index = b;
            }
        }
        if (active && (min_loss_index == loader->label[i])) {
            correct += 1;
        }
    }
    return correct;
}

void evalloader_free(EvalLoader *loader) {
    free(loader->buffer);
    free(loader->inputs);
    free(loader->targets);
    free(loader->mask);
    free(loader->label);
    fcloseCheck(loader->eval_file);
}

#endif // DATALOADER_H
