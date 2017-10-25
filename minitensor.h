// A small wrapper class for Torch tensors.

#ifndef minitensor__
#define minitensor__

extern "C" {
#include <assert.h>
}
#include <TH/TH.h>
#include <TH/THTensorMacros.h>
#include <string>
#include <sstream>
#include <fstream>

#ifdef SAFE
#define CHECK(expr) \
    do{if(!(expr)) abort();}while(0)
#define RCHECK(i, n) \
    do{if(unsigned(i) >= unsigned(n)) abort();}while(0)
#else
#define CHECK(expr) \
    do{}while(0)
#define RCHECK(i, n)                            \
    do{}while(0)
#endif

namespace {
#if 0
    std::ofstream memlog("memlog");
    void mlog(THFloatTensor *p, const char *msg, int line, const char *note) {
        std::stringstream entry;
        entry
            << std::dec << msg << " " << line << " " << note << " "
            << std::hex << (void*)p << " "
            << std::dec << p->refcount << std::endl;
        memlog << entry.str();
    }
#define M mlog(p, __FUNCTION__, __LINE__, note)
#else
#define M
#endif
}

class TFloat {
public:
    THFloatTensor *p = 0;
    const char *note = "_";
    TFloat() {
        p = THFloatTensor_new(); M;
    }
    TFloat(TFloat &other) {
        p = THFloatTensor_newWithTensor(other.p); M;
    }
    TFloat(const TFloat &other) {
        THFloatTensor_newWithTensor(other.p); M;
    }
    TFloat(THFloatTensor *p, const char *note) : p(p), note(note) {
        THFloatTensor_retain(p); M;
    }
    TFloat(THFloatTensor *p, bool retain, const char *note) : p(p), note(note) {
        if(retain) THFloatTensor_retain(p); M;
    }
    TFloat(int x0) {
        p = THFloatTensor_newWithSize1d(x0); M;
    }
    TFloat(int x0, int x1) {
        p = THFloatTensor_newWithSize2d(x0, x1); M;
    }
    TFloat(int x0, int x1, int x2) {
        p = THFloatTensor_newWithSize3d(x0, x1, x2); M;
    }
    TFloat(int x0, int x1, int x2, int x3) {
        p = THFloatTensor_newWithSize4d(x0, x1, x2, x3); M;
    }
    ~TFloat() {
        assert(p); M;
        THFloatTensor_free(p);
        p = 0;
    }
    std::string info() {
        std::stringstream result;
        for (int i=0; i<dim(); i++)
            result << " " << size(i);
        result << " " << min();
        result << " " << max();
        return result.str();
    }
    operator THFloatTensor *() {
        return p;
    }
    bool isSameSizeAs(TFloat &other) {
        return THFloatTensor_isSameSizeAs(p, other.p);
    }
    TFloat &operator=(const TFloat &other) {
        THFloatTensor_retain(other.p);
        THFloatTensor_free(p);
        p = other.p;
        return *this;
    }
    TFloat &operator=(THFloatTensor *other_p) {
        THFloatTensor_retain(other_p);
        THFloatTensor_free(p);
        p = other_p;
        return *this;
    }
    TFloat &operator=(float value) {
        THFloatTensor_fill(p, value);
        return *this;
    }
    TFloat &assign(TFloat &other) {
        THFloatTensor_resizeAs(p, other.p);
        THFloatTensor_copy(p, other.p);
        return *this;
    }
    TFloat &copy(TFloat &other) {
        THFloatTensor_copy(p, other.p);
        return *this;
    }
    TFloat &fill(float value) {
        THFloatTensor_fill(p, value);
        return *this;
    }
    TFloat &zero() {
        THFloatTensor_zero(p);
        return *this;
    }

    TFloat narrow(int dimension, int start, int size) {
        TFloat result(THFloatTensor_newNarrow(p, dimension, start, size), false, "narrow");
        return result;
    }
    TFloat select(int dimension, int index) {
        TFloat result(THFloatTensor_newSelect(p, dimension, index), false, "select");
        return result;
    }
    TFloat squeeze() {
        TFloat result;
        THFloatTensor_squeeze(result.p, p);
        return result;
    }
    TFloat squeeze(int dimension) {
        TFloat result;
        THFloatTensor_squeeze1d(result.p, p, dimension);
        return result;
    }
#if 0
    TFloat unsqueeze(int dimension) {
        TFloat result;
        THFloatTensor_unsqueeze1d(result.p, p, dimension);
        return result;
    }
#endif
    TFloat transpose(int dim1, int dim2) {
        TFloat result(THFloatTensor_newTranspose(p, dim1, dim2), false, "transpose");
        return result;
    }

    TFloat &resize(int x0) {
        THFloatTensor_resize1d(p, x0);
        return *this;
    }
    TFloat &resize(int x0, int x1) {
        THFloatTensor_resize2d(p, x0, x1);
        return *this;
    }
    TFloat &resize(int x0, int x1, int x2) {
        THFloatTensor_resize3d(p, x0, x1, x2);
        return *this;
    }
    TFloat &resize(int x0, int x1, int x2, int x3) {
        THFloatTensor_resize4d(p, x0, x1, x2, x3);
        return *this;
    }

    int dim() {
        return THFloatTensor_nDimension(p);
    }
    int size(int i) {
        return THFloatTensor_size(p, i);
    }

    float &operator()(int x0) {
        CHECK(dim()==1);
        RCHECK(x0, size(0));
        return THTensor_fastGet1d(p, x0);
    }
    float &operator()(int x0, int x1) {
        CHECK(dim()==2);
        RCHECK(x0, size(0));
        RCHECK(x1, size(1));
        return THTensor_fastGet2d(p, x0, x1);
    }
    float &operator()(int x0, int x1, int x2) {
        CHECK(dim()==3);
        RCHECK(x0, size(0));
        RCHECK(x1, size(1));
        RCHECK(x2, size(2));
        return THTensor_fastGet3d(p, x0, x1, x2);
    }
    float &operator()(int x0, int x1, int x2, int x3) {
        CHECK(dim()==4);
        RCHECK(x0, size(0));
        RCHECK(x1, size(1));
        RCHECK(x2, size(2));
        RCHECK(x3, size(3));
        return THTensor_fastGet4d(p, x0, x1, x2, x3);
    }

    TFloat &resizeAs(TFloat &other) {
        THFloatTensor_resizeAs(p, other.p);
        return *this;
    }
    TFloat &operator+=(TFloat &other) {
        THFloatTensor_add(p, other.p, 1.0);
        return *this;
    }
    TFloat &cadd(TFloat &input1, float c, TFloat &input2) {
        THFloatTensor_cadd(p, input1.p, c, input2.p);
        return *this;
    }

    float min() {
        return THFloatTensor_minall(p);
    }
    float max() {
        return THFloatTensor_maxall(p);
    }
    float sum() {
        return THFloatTensor_sumall(p);
    }
    float prod() {
        return THFloatTensor_prodall(p);
    }

    TFloat operator+(float x) {
        TFloat result;
        THFloatTensor_add(result.p, p, x);
        return result;
    }
    TFloat operator-(float x) {
        TFloat result;
        THFloatTensor_add(result.p, p, -x);
        return result;
    }
    TFloat operator*(float x) {
        TFloat result;
        THFloatTensor_mul(result.p, p, x);
        return result;
    }
    TFloat operator/(float x) {
        TFloat result;
        THFloatTensor_div(result.p, p, x);
        return result;
    }
    TFloat clamp(float x, float y) {
        TFloat result;
        THFloatTensor_clamp(result.p, p, x, y);
        return result;
    }

    TFloat operator+(TFloat &other) {
        TFloat result;
        THFloatTensor_cadd(result.p, p, 1.0, other.p);
        return result;
    }
    TFloat operator-(TFloat &other) {
        TFloat result;
        THFloatTensor_cadd(result.p, p, -1.0, other.p);
        return result;
    }
    TFloat operator*(TFloat &other) {
        TFloat result;
        THFloatTensor_cmul(result.p, p, other.p);
        return result;
    }
    TFloat operator/(TFloat &other) {
        TFloat result;
        THFloatTensor_cdiv(result.p, p, other.p);
        return result;
    }
};

#endif
