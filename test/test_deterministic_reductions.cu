#include "cuda/aux/gdn.cuh"
#include "engine/glue2.cuh"
#include "engine/glue.cuh"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cstring>
using namespace helios;
static void check(cudaError_t e) { if(e!=cudaSuccess) {fprintf(stderr,"%s\n",cudaGetErrorString(e));std::exit(1);} }
template<class T> struct Buffer {
  T* p; size_t n;
  Buffer(size_t count):n(count) {check(cudaMalloc(&p,n*sizeof(T)));}
  ~Buffer(){cudaFree(p);}
  void put(const std::vector<T>& v){check(cudaMemcpy(p,v.data(),n*sizeof(T),cudaMemcpyHostToDevice));}
  std::vector<T> get(){std::vector<T> v(n);check(cudaMemcpy(v.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost));return v;}
};
static void require(bool pass,const char* what){if(!pass){fprintf(stderr,"FAIL %s\n",what);std::exit(1);}}
static void recurrence(int dim,bool channelwise,bool history,int steps) {
  using BF=aux::bfloat16;
  const int heads=2,stride=history?steps+1:1;
  Buffer<BF> qkv(steps*3*heads*dim),beta(steps*heads),output(steps*heads*dim);
  Buffer<float> decay(steps*heads*(channelwise?dim:1)),state(stride*heads*dim*dim);
  std::vector<BF> q(qkv.n),b(beta.n);
  std::vector<float> g(decay.n),initial(state.n);
  for(size_t i=0;i<q.size();++i) q[i]=__float2bfloat16_rn(std::sin(i*.019f));
  for(size_t i=0;i<b.size();++i) b[i]=__float2bfloat16_rn(.3f+std::cos(i)*.1f);
  for(size_t i=0;i<g.size();++i) g[i]=-.01f-(i%7)*.003f;
  for(size_t i=0;i<initial.size();++i) initial[i]=std::sin(i*.017f)*.2f;
  qkv.put(q);beta.put(b);decay.put(g);
  std::vector<float> reference_state;std::vector<BF> reference_output;
  for(int repeat=0;repeat<16;++repeat) {
    state.put(initial);
    aux::cuda_recurrent_gated_delta_rule(qkv.p,decay.p,beta.p,state.p,output.p,
      1,steps,heads,heads,dim,dim,stride,nullptr,channelwise,history,nullptr);
    check(cudaDeviceSynchronize());auto s=state.get();auto o=output.get();
    if(!repeat){reference_state=s;reference_output=o;}
    require(!std::memcmp(s.data(),reference_state.data(),s.size()*sizeof(float)),"recurrence state bytes");
    require(!std::memcmp(o.data(),reference_output.data(),o.size()*sizeof(BF)),"recurrence output bytes");
  }
  printf("PASS recurrence dim=%d channelwise=%d history=%d steps=%d repeats=16 zero differing bytes\n",dim,channelwise,history,steps);
}
int main() {
  check(cudaSetDevice(0));
  for(int dim:{64,128}) for(bool channelwise:{false,true}) {
    if(channelwise&&dim!=128) continue;
    for(bool history:{false,true}) for(int steps:{1,7}) recurrence(dim,channelwise,history,steps);
  }
  // Contribution buffer order may change with the integer permutation cursor. The
  // inverse map must recover the identical routing order, including cancellation.
  const int hidden=513,topk=4;
  Buffer<float> contributions(topk*hidden),result(hidden);
  Buffer<int64_t> inverse(topk);
  const float terms[]={1e20f,-1e20f,1.f,1.f};
  for(int repeat=0;repeat<8;++repeat) {
    std::vector<float> c(contributions.n);std::vector<int64_t> map(topk);
    for(int j=0;j<topk;++j){int row=(j+repeat)%topk;map[j]=row;for(int d=0;d<hidden;++d)c[row*hidden+d]=terms[j];}
    contributions.put(c);inverse.put(map);
    glue::moe_reduce_sorted(result.p,contributions.p,inverse.p,1,topk,hidden,nullptr);
    check(cudaDeviceSynchronize());for(float v:result.get())require(v==2.f,"MoE fixed routing order");
  }
  puts("PASS MoE fixed routing order through 8 permutations, fp32 cancellation result=2");
  Buffer<float> src(2*hidden),y(3*hidden);Buffer<int64_t> idx(2);Buffer<half> weights(2);
  src.put(std::vector<float>(src.n,3.f));y.put(std::vector<float>(y.n,1.f));idx.put({2,0});weights.put({__float2half_rn(.5f),__float2half_rn(.25f)});
  for(int r=0;r<3;++r)glue::scatter_add_rows(y.p,src.p,idx.p,weights.p,2,hidden,nullptr);
  auto v=y.get();for(int d=0;d<hidden;++d)require(v[d]==3.25f&&v[hidden+d]==1.f&&v[2*hidden+d]==5.5f,"unique scatter accumulation");
  Buffer<half> x(6),w(15);Buffer<float> gemm(10);
  x.put(std::vector<half>(6,__float2half_rn(2.f)));w.put(std::vector<half>(15,__float2half_rn(.5f)));gemm.put(std::vector<float>(10,1.f));
  for(int r=0;r<3;++r)glue::gemm_nt_f16(gemm.p,x.p,w.p,2,5,3,true,true,nullptr);
  for(float z:gemm.get())require(z==10.f,"gemm single-owner addition");
  puts("PASS unique scatter and tiled GEMM additions");
}
