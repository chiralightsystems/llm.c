#include "llmc/eos_boundary.h"
#include <cstdio>
#include <set>

static int failures=0;
static void check(bool ok,const char* label){if(!ok){std::fprintf(stderr,"FAIL %s\n",label);++failures;}}
static void verify(const std::vector<int>& tokens,int B,int T,int eos) {
    const auto original=tokens;
    const auto p=llmc_make_eos_plan(tokens.data(),B,T,128,eos);
    std::vector<int> covered(tokens.size(),0);
    for(const auto& g:p.groups) {
        check(g.count<=LLMC_EOS_GROUP_LIMIT&&g.count*g.max_sequence<=std::max(T,LLMC_EOS_TOKEN_BUDGET),"execution resource bounds");
        for(int i=0;i<g.count;++i) {
            size_t n=g.metadata_begin+i;
            int len=p.lengths[n];
            const int begin=(int)(p.output_offsets[n]/128);
            if(!len){check(begin==B*T,"padding terminal offset");continue;}
            check(len<=g.max_sequence&&begin/T==(begin+len-1)/T,"segment contained in row");
            check(begin%T==0||tokens[begin]==eos,"segment begins at row or EOS input");
            for(int t=0;t<len;++t){++covered[begin+t];check(p.positions[begin+t]==t,"local RoPE position");if(t)check(tokens[begin+t]!=eos,"no interior EOS");}
        }
    }
    for(int n:covered)check(n==1,"every token covered exactly once");
    check(tokens==original,"input order/targets untouched");
    for(int b=0;b<B;++b)for(int t=0;t<T-1;++t) {
        const int target=tokens[b*T+t+1];
        if(target==eos)check(p.positions[b*T+t+1]==0,"scored EOS target resets its own input only");
    }
}
int main() {
    int p=-1;
    check(llmc_parse_attention_boundary("row_causal_v1",&p)&&p==0,"legacy policy parser");
    check(llmc_parse_attention_boundary("isolate_segments_v1",&p)&&p==1,"isolated parser");
    check(!llmc_parse_attention_boundary("row_reset",&p),"attention policy separate from row loss");
    check(llmc_parse_eos_id("50256",&p)&&p==50256&&llmc_parse_eos_id("-1",&p)&&p==-1,"EOS integer parser");
    for(const char* text:{"", "x", "127x", "-2", "922337203685477580899"})check(!llmc_parse_eos_id(text,&p),"invalid EOS integer rejected");
    check(llmc_valid_eos_policy(0,-1,128)&&!llmc_valid_eos_policy(0,127,128),"legacy token contract");
    check(llmc_valid_eos_policy(1,127,128)&&!llmc_valid_eos_policy(1,128,128),"EOS range");
    int h[256]={};llmc_store_eos_contract(h,15,127);
    check(llmc_eos_contract_matches(h,true,1,127,128),"model EOS contract roundtrip");
    check(!llmc_eos_contract_matches(h,true,0,-1,128)&&!llmc_eos_contract_matches(h,true,1,126,128),"cross policy/token reject");
    check(!llmc_eos_contract_matches(h,false,1,127,128),"legacy resume cannot enable EOS");
    h[65]=2;check(!llmc_valid_eos_contract(h,128),"unknown schema rejected");
    verify({1,2,3,4,5,6,7,8,9,10,11,12},2,6,127);
    verify({127,1,127,127,4,127,5,6,127,9,10,11},2,6,127);
    verify(std::vector<int>(21,127),3,7,127);
    verify({127},1,1,127);
    std::vector<int> dense(32769,127);verify(dense,1,(int)dense.size(),127);
    // Exhaust all boundary partitions of two unequal lanes through length8.
    for(int mask=0;mask<256;++mask){std::vector<int> x(16,1);for(int t=0;t<8;++t){if(mask&(1<<t))x[t]=127;if((255-mask)&(1<<t))x[8+t]=127;}verify(x,2,8,127);}
    bool rejected=false;try{llmc_make_eos_plan(nullptr,1,8,128,127);}catch(const std::invalid_argument&){rejected=true;}
    check(rejected,"invalid plan rejected");
    std::printf("llmc_eos_boundary_host: result=%s\n",failures?"fail":"pass");return failures?1:0;
}
