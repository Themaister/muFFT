
ifeq ($(PLATFORM),)
   PLATFORM = unix
   ifeq ($(shell uname -a),)
      PLATFORM = win
   else ifneq ($(findstring MINGW,$(shell uname -a)),)
      PLATFORM = win
   else ifneq ($(findstring Darwin,$(shell uname -a)),)
      PLATFORM = osx
   else ifneq ($(findstring win,$(shell uname -a)),)
      PLATFORM = win
   endif
endif

CFLAGS += -std=gnu99 -Wall -Wextra -pedantic
LDFLAGS += -lm -Wl,-no-undefined

ifeq ($(PLATFORM),win)
   CC = $(TOOLCHAIN_PREFIX)gcc
   AR = $(TOOLCHAIN_PREFIX)ar
   EXE_SUFFIX := .exe
   SHARED := -shared
   TARGET_SHARED := mufft.dll
   TARGET_STATIC := libmufft.a
else ifeq ($(PLATFORM),osx)
   FPIC := -fPIC
   SHARED := -dynamiclib
   TARGET_SHARED := libmufft.dylib
   TARGET_STATIC := libmufft.a
else
   FPIC := -fPIC
   SHARED := -shared
   TARGET_SHARED := libmufft.so
   TARGET_STATIC := libmufft.a
endif

ifneq ($(TOOLCHAIN_PREFIX),)
   CC = $(TOOLCHAIN_PREFIX)gcc
   AR = $(TOOLCHAIN_PREFIX)ar
endif

ifeq ($(ARCH),)
   ARCH := $(shell $(CC) -dumpmachine)
endif

ifeq ($(DEBUG), 1)
   CONFIG := debug
   CFLAGS += -O0 -g -DMUFFT_DEBUG
else
   CONFIG := release
   CFLAGS += -Ofast
endif

ifeq ($(SANITIZE), 1)
   CC = clang
   CFLAGS += -fsanitize=memory
   LDFLAGS += -fsanitize=memory
endif

PLATFORM_SOURCES_X86 := x86/kernel.sse.c x86/kernel.sse3.c x86/kernel.avx.c
PLATFORM_DEFINES_X86 := -DMUFFT_HAVE_SSE -DMUFFT_HAVE_SSE3 -DMUFFT_HAVE_AVX
ifneq ($(findstring x86_64,$(ARCH)),)
   PLATFORM_SOURCES := $(PLATFORM_SOURCES_X86) 
   PLATFORM_DEFINES := $(PLATFORM_DEFINES_X86) 
   PLATFORM_ARCH := x86_64
else ifneq ($(findstring i386,$(ARCH)),)
   PLATFORM_SOURCES := $(PLATFORM_SOURCES_X86) 
   PLATFORM_DEFINES := $(PLATFORM_DEFINES_X86) 
   PLATFORM_ARCH := x86
else ifneq ($(findstring i486,$(ARCH)),)
   PLATFORM_SOURCES := $(PLATFORM_SOURCES_X86) 
   PLATFORM_DEFINES := $(PLATFORM_DEFINES_X86) 
   PLATFORM_ARCH := x86
else ifneq ($(findstring i586,$(ARCH)),)
   PLATFORM_SOURCES := $(PLATFORM_SOURCES_X86) 
   PLATFORM_DEFINES := $(PLATFORM_DEFINES_X86) 
   PLATFORM_ARCH := x86
else ifneq ($(findstring i686,$(ARCH)),)
   PLATFORM_SOURCES := $(PLATFORM_SOURCES_X86) 
   PLATFORM_DEFINES := $(PLATFORM_DEFINES_X86) 
   PLATFORM_ARCH := x86
endif

CFLAGS += $(PLATFORM_DEFINES)

OBJDIR_SHARED := obj-shared/$(PLATFORM_ARCH)/$(PLATFORM)/$(CONFIG)
OBJDIR_STATIC := obj-static/$(PLATFORM_ARCH)/$(PLATFORM)/$(CONFIG)
TARGET_OUT_SHARED := bin/$(PLATFORM_ARCH)/$(PLATFORM)/$(CONFIG)/$(TARGET_SHARED)
TARGET_OUT_STATIC := bin/$(PLATFORM_ARCH)/$(PLATFORM)/$(CONFIG)/$(TARGET_STATIC)
TARGET_TEST := mufft_test$(EXE_SUFFIX)
SOURCES := fft.c kernel.c
SOURCES_TEST := test.c

OBJECTS_SHARED := \
	$(addprefix $(OBJDIR_SHARED)/,$(SOURCES:.c=.o)) \
	$(addprefix $(OBJDIR_SHARED)/,$(PLATFORM_SOURCES:.c=.o))

OBJECTS_STATIC := \
	$(addprefix $(OBJDIR_STATIC)/,$(SOURCES:.c=.o)) \
	$(addprefix $(OBJDIR_STATIC)/,$(PLATFORM_SOURCES:.c=.o))

OBJECTS_TEST := \
	$(addprefix $(OBJDIR_STATIC)/,$(SOURCES_TEST:.c=.o))

DEPS := $(OBJECTS_SHARED:.o=.d) $(OBJECTS_STATIC:.o=.d) $(OBJECTS_TEST:.o=.d)

all: static shared

static: $(TARGET_OUT_STATIC)

shared: $(TARGET_OUT_SHARED)

test: $(TARGET_TEST)

-include $(DEPS)

$(TARGET_TEST): static $(OBJECTS_TEST)
	$(CC) -o $@ $(OBJECTS_TEST) $(TARGET_OUT_STATIC) $(shell pkg-config fftw3f --libs) $(LDFLAGS) 

$(TARGET_OUT_SHARED): $(OBJECTS_SHARED)
	@mkdir -p $(dir $@)
	$(CC) -o $@ $^ $(LDFLAGS) $(FPIC) $(SHARED)

$(TARGET_OUT_STATIC): $(OBJECTS_STATIC)
	@mkdir -p $(dir $@)
	$(AR) rcs $@ $^

$(OBJDIR_SHARED)/%.sse.o: %.c
	@mkdir -p $(dir $@)
	$(CC) -c -o $@ $< $(CFLAGS) -MMD -msse -mno-sse3 -mno-avx $(FPIC)

$(OBJDIR_SHARED)/%.sse3.o: %.c
	@mkdir -p $(dir $@)
	$(CC) -c -o $@ $< $(CFLAGS) -MMD -msse -msse3 -mno-avx $(FPIC)

$(OBJDIR_SHARED)/%.avx.o: %.c
	@mkdir -p $(dir $@)
	$(CC) -c -o $@ $< $(CFLAGS) -MMD -msse -msse3 -mavx $(FPIC)

$(OBJDIR_SHARED)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) -c -o $@ $< $(CFLAGS) -MMD $(FPIC)

$(OBJDIR_STATIC)/%.sse.o: %.c
	@mkdir -p $(dir $@)
	$(CC) -c -o $@ $< $(CFLAGS) -MMD -msse -mno-sse3 -mno-avx

$(OBJDIR_STATIC)/%.sse3.o: %.c
	@mkdir -p $(dir $@)
	$(CC) -c -o $@ $< $(CFLAGS) -MMD -msse -msse3 -mno-avx

$(OBJDIR_STATIC)/%.avx.o: %.c
	@mkdir -p $(dir $@)
	$(CC) -c -o $@ $< $(CFLAGS) -MMD -msse -msse3 -mavx

$(OBJDIR_STATIC)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) -c -o $@ $< $(CFLAGS) -MMD

clean:
	rm -f $(TARGET_OUT_SHARED) $(TARGET_OUT_STATIC) $(TARGET_TEST)
	rm -rf $(OBJDIR_SHARED) $(OBJDIR_STATIC)

clean-all:
	rm -rf bin obj-shared obj-static

.PHONY: all test shared static clean clean-all

