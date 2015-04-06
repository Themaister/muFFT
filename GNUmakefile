
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

ifeq ($(PLATFORM),win)
   CC = $(TOOLCHAIN_PREFIX)gcc
   EXE_SUFFIX := .exe
   SHARED := -shared
else
   FPIC := -fPIC
   SHARED := -shared
endif

CFLAGS += -std=c99 -Wall -Wextra -pedantic $(FPIC) -D_POSIX_C_SOURCE=200112L
LDFLAGS += -lm $(SHARED) -Wl,-no-undefined

ifneq ($(TOOLCHAIN_PREFIX),)
   CC = $(TOOLCHAIN_PREFIX)gcc
endif

ifeq ($(ARCH),)
   ARCH := $(shell $(CC) -dumpmachine)
endif

ifeq ($(DEBUG), 1)
   CONFIG := debug
   CFLAGS += -O0 -g
else
   CONFIG := release
   CFLAGS += -Ofast
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

TARGET_SHARED := libmufft.so
OBJDIR := obj/$(PLATFORM_ARCH)/$(CONFIG)
TARGET := bin/$(PLATFORM_ARCH)/$(CONFIG)/$(TARGET_SHARED)
SOURCES := fft.c kernel.c

OBJECTS := \
	$(addprefix $(OBJDIR)/,$(SOURCES:.c=.o)) \
	$(addprefix $(OBJDIR)/,$(PLATFORM_SOURCES:.c=.o))

DEPS := $(OBJECTS:.o=.d)

all: $(TARGET)

-include $(DEPS)

$(TARGET): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(CC) -o $@ $^ $(LDFLAGS) $(FPIC)

$(OBJDIR)/%.sse.o: %.c
	@mkdir -p $(dir $@)
	$(CC) -c -o $@ $< $(CFLAGS) -MMD -msse -mno-sse3 -mno-avx

$(OBJDIR)/%.sse3.o: %.c
	@mkdir -p $(dir $@)
	$(CC) -c -o $@ $< $(CFLAGS) -MMD -msse -msse3 -mno-avx

$(OBJDIR)/%.avx.o: %.c
	@mkdir -p $(dir $@)
	$(CC) -c -o $@ $< $(CFLAGS) -MMD -msse -msse3 -mavx

$(OBJDIR)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) -c -o $@ $< $(CFLAGS) -MMD

clean:
	rm -f $(TARGET)
	rm -rf $(OBJDIR)

clean-all:
	rm -rf bin obj

.PHONY: clean clean-all

