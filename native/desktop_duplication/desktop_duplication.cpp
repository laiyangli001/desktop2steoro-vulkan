#include "desktop_duplication.h"
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d11.h>
#include <dxgi1_2.h>
#include <wrl/client.h>
#include <cstring>
#include <string>
using Microsoft::WRL::ComPtr;
namespace {
thread_local std::string error_text;
void error(const char* s){ error_text=s?s:"unknown Desktop Duplication error"; }
struct State { ComPtr<IDXGIAdapter1> adapter; ComPtr<ID3D11Device> device; ComPtr<ID3D11DeviceContext> context; ComPtr<IDXGIOutputDuplication> duplication; ComPtr<IDXGIResource> resource; ComPtr<ID3D11Texture2D> texture; ComPtr<ID3D11Texture2D> staging; LUID luid{}; int output_index=0; int timeout=16; bool acquired=false; };
void release(State* s){ if(!s)return; if(s->acquired&&s->duplication)s->duplication->ReleaseFrame(); s->acquired=false; s->texture.Reset(); s->resource.Reset(); }
bool create(int wanted, State* s){
 if(!s||wanted<0){error("invalid output index");return false;} release(s); s->duplication.Reset(); s->adapter.Reset(); s->device.Reset(); s->context.Reset(); s->output_index=wanted; ComPtr<IDXGIFactory1> f; HRESULT hr=CreateDXGIFactory1(IID_PPV_ARGS(&f)); if(FAILED(hr)){error("CreateDXGIFactory1 failed");return false;} int n=0;
 for(UINT ai=0;;++ai){ ComPtr<IDXGIAdapter1>a; if(f->EnumAdapters1(ai,&a)==DXGI_ERROR_NOT_FOUND)break; DXGI_ADAPTER_DESC1 ad{}; if(FAILED(a->GetDesc1(&ad))||(ad.Flags&DXGI_ADAPTER_FLAG_SOFTWARE))continue;
  for(UINT oi=0;;++oi){ ComPtr<IDXGIOutput>o; if(a->EnumOutputs(oi,&o)==DXGI_ERROR_NOT_FOUND)break; DXGI_OUTPUT_DESC od{}; if(FAILED(o->GetDesc(&od))||!od.AttachedToDesktop)continue; if(n++!=wanted)continue;
   ComPtr<ID3D11Device>d; ComPtr<ID3D11DeviceContext>c; D3D_FEATURE_LEVEL fl{}; const D3D_FEATURE_LEVEL levels[]={D3D_FEATURE_LEVEL_11_1,D3D_FEATURE_LEVEL_11_0,D3D_FEATURE_LEVEL_10_1}; hr=D3D11CreateDevice(a.Get(),D3D_DRIVER_TYPE_UNKNOWN,nullptr,D3D11_CREATE_DEVICE_BGRA_SUPPORT,levels,3,D3D11_SDK_VERSION,&d,&fl,&c); if(FAILED(hr)){error("D3D11CreateDevice failed");return false;}
   ComPtr<IDXGIOutput1>o1; if(FAILED(o.As(&o1))){error("IDXGIOutput1 unavailable");return false;} ComPtr<IDXGIOutputDuplication>dup; hr=o1->DuplicateOutput(d.Get(),&dup); if(FAILED(hr)){error(hr==DXGI_ERROR_UNSUPPORTED?"Desktop Duplication unsupported by adapter/driver":"DuplicateOutput failed");return false;}
   s->adapter=a;s->device=d;s->context=c;s->duplication=dup;s->luid=ad.AdapterLuid;return true;
  }
 }
 error("no attached DXGI output found"); return false;
}
}
extern "C" D2S_DESKTOP_DUPLICATION_API int d2s_desktop_duplication_probe(){error_text.clear();State s;return create(0,&s)?1:0;}
extern "C" D2S_DESKTOP_DUPLICATION_API int d2s_desktop_duplication_last_error(char*out,int cap){if(!out||cap<=0)return(int)error_text.size();int n=(int)error_text.size(),m=n<cap-1?n:cap-1;std::memcpy(out,error_text.data(),m);out[m]='\0';return n;}
extern "C" D2S_DESKTOP_DUPLICATION_API void* d2s_desktop_duplication_create(int output,int timeout){error_text.clear();auto*s=new State();s->timeout=timeout>0?timeout:16;if(!create(output,s)){delete s;return nullptr;}return s;}
extern "C" D2S_DESKTOP_DUPLICATION_API int d2s_desktop_duplication_acquire(void*h,void**out,int*w,int*hh,uint64_t*l){auto*s=(State*)h;if(!s||!out||!w||!hh||!l||!s->duplication){error("Desktop Duplication session is not initialized");return-1;}if(s->acquired){error("previous frame not released");return-1;}DXGI_OUTDUPL_FRAME_INFO fi{};ComPtr<IDXGIResource>r;HRESULT hr=s->duplication->AcquireNextFrame((UINT)s->timeout,&fi,&r);if(hr==DXGI_ERROR_WAIT_TIMEOUT)return 0;if(hr==DXGI_ERROR_ACCESS_LOST){
 error("Desktop Duplication access lost; recreating output");
 if(!create(s->output_index,s)){return-1;}
 return-2;
}if(FAILED(hr)){error("AcquireNextFrame failed");return-1;}ComPtr<ID3D11Texture2D>t;if(FAILED(r.As(&t))){s->duplication->ReleaseFrame();error("acquired resource is not ID3D11Texture2D");return-1;}D3D11_TEXTURE2D_DESC d{};t->GetDesc(&d);s->resource=r;s->texture=t;s->acquired=true;*out=t.Get();*w=(int)d.Width;*hh=(int)d.Height;*l=((uint64_t)(uint32_t)s->luid.HighPart<<32)|(uint32_t)s->luid.LowPart;return 1;}
extern "C" D2S_DESKTOP_DUPLICATION_API int d2s_desktop_duplication_copy_frame(void*h,unsigned char*out,int capacity,int*stride,int*w,int*hh){auto*s=(State*)h;if(!s||!s->acquired||!s->texture||!s->device||!s->context||!stride||!w||!hh){error("no acquired Desktop Duplication frame for readback");return-1;}D3D11_TEXTURE2D_DESC d{};s->texture->GetDesc(&d);int row_bytes=(int)(d.Width*4),required=row_bytes*(int)d.Height;*stride=row_bytes;*w=(int)d.Width;*hh=(int)d.Height;if(!out||capacity<required){error("Desktop Duplication readback buffer is too small");return-2;}if(!s->staging){D3D11_TEXTURE2D_DESC sd=d;sd.Usage=D3D11_USAGE_STAGING;sd.BindFlags=0;sd.CPUAccessFlags=D3D11_CPU_ACCESS_READ;sd.MiscFlags=0;HRESULT create_hr=s->device->CreateTexture2D(&sd,nullptr,&s->staging);if(FAILED(create_hr)){error("CreateTexture2D staging readback failed");return-1;}}s->context->CopyResource(s->staging.Get(),s->texture.Get());s->context->Flush();D3D11_MAPPED_SUBRESOURCE mapped{};HRESULT map_hr=s->context->Map(s->staging.Get(),0,D3D11_MAP_READ,0,&mapped);if(FAILED(map_hr)){error("Map Desktop Duplication staging readback failed");return-1;}for(UINT y=0;y<d.Height;++y){std::memcpy(out+(size_t)y*row_bytes,(const unsigned char*)mapped.pData+(size_t)y*mapped.RowPitch,row_bytes);}s->context->Unmap(s->staging.Get(),0);return required;}
extern "C" D2S_DESKTOP_DUPLICATION_API void* d2s_desktop_duplication_device(void*h){auto*s=(State*)h;return s&&s->device?s->device.Get():nullptr;}
extern "C" D2S_DESKTOP_DUPLICATION_API int d2s_desktop_duplication_release_frame(void*h){auto*s=(State*)h;if(!s){error("invalid handle");return-1;}release(s);return 1;}
extern "C" D2S_DESKTOP_DUPLICATION_API void d2s_desktop_duplication_destroy(void*h){auto*s=(State*)h;if(!s)return;release(s);delete s;}
#endif
