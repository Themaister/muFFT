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

CP := cp
MKDIR := mkdir
INSTALL := install
SED := sed
RM := rm
DOXYGEN := doxygen

PKGCONF_FILE := mufft.pc
VERSION := 0.1

PREFIX = /usr/local

CFLAGS += -std=gnu99 -Wall -Wextra
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

SIMD_ENABLE = 1

ifeq ($(SIMD_ENABLE), 1)

PLATFORM_SOURCES_X86 := x86/kernel.sse.c x86/kernel.sse3.c x86/kernel.avx.c
PLATFORM_DEFINES_X86 := -DMUFFT_HAVE_SSE -DMUFFT_HAVE_SSE3 -DMUFFT_HAVE_AVX -DMUFFT_HAVE_X86
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
endif

CFLAGS += $(PLATFORM_DEFINES)

OBJDIR_SHARED := obj-shared/$(PLATFORM_ARCH)/$(PLATFORM)/$(CONFIG)
OBJDIR_STATIC := obj-static/$(PLATFORM_ARCH)/$(PLATFORM)/$(CONFIG)
TARGET_OUT_SHARED := bin/$(PLATFORM_ARCH)/$(PLATFORM)/$(CONFIG)/$(TARGET_SHARED)
TARGET_OUT_STATIC := bin/$(PLATFORM_ARCH)/$(PLATFORM)/$(CONFIG)/$(TARGET_STATIC)
TARGET_TEST := mufft_test$(EXE_SUFFIX)
TARGET_BENCH := mufft_bench$(EXE_SUFFIX)
SOURCES := fft.c kernel.c cpu.c
SOURCES_TEST := test.c
SOURCES_BENCH := bench.c

OBJECTS_SHARED := \
	$(addprefix $(OBJDIR_SHARED)/,$(SOURCES:.c=.o)) \
	$(addprefix $(OBJDIR_SHARED)/,$(PLATFORM_SOURCES:.c=.o))

OBJECTS_STATIC := \
	$(addprefix $(OBJDIR_STATIC)/,$(SOURCES:.c=.o)) \
	$(addprefix $(OBJDIR_STATIC)/,$(PLATFORM_SOURCES:.c=.o))

OBJECTS_TEST := \
	$(addprefix $(OBJDIR_STATIC)/,$(SOURCES_TEST:.c=.o))

OBJECTS_BENCH := \
	$(addprefix $(OBJDIR_STATIC)/,$(SOURCES_BENCH:.c=.o))

DEPS := $(OBJECTS_SHARED:.o=.d) $(OBJECTS_STATIC:.o=.d) $(OBJECTS_TEST:.o=.d)

-include $(DEPS)

all: static shared

static: $(TARGET_STATIC)

shared: $(TARGET_SHARED)

test: $(TARGET_TEST)

bench: $(TARGET_BENCH)


$(TARGET_TEST): static $(OBJECTS_TEST)
	$(CC) -o $@ $(OBJECTS_TEST) $(TARGET_OUT_STATIC) $(shell pkg-config fftw3f --libs) $(LDFLAGS)

$(TARGET_BENCH): static $(OBJECTS_BENCH)
	$(CC) -o $@ $(OBJECTS_BENCH) $(TARGET_OUT_STATIC) $(shell pkg-config fftw3f --libs) $(LDFLAGS)

$(TARGET_SHARED): $(TARGET_OUT_SHARED)
	$(CP) $< $@

$(TARGET_STATIC): $(TARGET_OUT_STATIC)
	$(CP) $< $@

$(TARGET_OUT_SHARED): $(OBJECTS_SHARED)
	@$(MKDIR) -p $(dir $@)
	$(CC) -o $@ $^ $(LDFLAGS) $(FPIC) $(SHARED)

$(TARGET_OUT_STATIC): $(OBJECTS_STATIC)
	@$(MKDIR) -p $(dir $@)
	$(AR) rcs $@ $^

$(OBJDIR_SHARED)/%.sse.o: %.c
	@$(MKDIR) -p $(dir $@)
	$(CC) -c -o $@ $< $(CFLAGS) -MMD -msse -mno-sse3 -mno-avx $(FPIC)

$(OBJDIR_SHARED)/%.sse3.o: %.c
	@$(MKDIR) -p $(dir $@)
	$(CC) -c -o $@ $< $(CFLAGS) -MMD -msse -msse3 -mno-avx $(FPIC)

$(OBJDIR_SHARED)/%.avx.o: %.c
	@$(MKDIR) -p $(dir $@)
	$(CC) -c -o $@ $< $(CFLAGS) -MMD -msse -msse3 -mavx $(FPIC)

$(OBJDIR_SHARED)/%.o: %.c
	@$(MKDIR) -p $(dir $@)
	$(CC) -c -o $@ $< $(CFLAGS) -MMD $(FPIC)

$(OBJDIR_STATIC)/%.sse.o: %.c
	@$(MKDIR) -p $(dir $@)
	$(CC) -c -o $@ $< $(CFLAGS) -MMD -msse -mno-sse3 -mno-avx

$(OBJDIR_STATIC)/%.sse3.o: %.c
	@$(MKDIR) -p $(dir $@)
	$(CC) -c -o $@ $< $(CFLAGS) -MMD -msse -msse3 -mno-avx

$(OBJDIR_STATIC)/%.avx.o: %.c
	@$(MKDIR) -p $(dir $@)
	$(CC) -c -o $@ $< $(CFLAGS) -MMD -msse -msse3 -mavx

$(OBJDIR_STATIC)/%.o: %.c
	@$(MKDIR) -p $(dir $@)
	$(CC) -c -o $@ $< $(CFLAGS) -MMD

docs:
	$(DOXYGEN)

clean:
	$(RM) -f $(TARGET_OUT_SHARED) $(TARGET_OUT_STATIC) $(TARGET_TEST) $(TARGET_BENCH)
	$(RM) -rf $(OBJDIR_SHARED) $(OBJDIR_STATIC)

clean-all:
	$(RM) -rf bin obj-shared obj-static $(PKGCONF_FILE)

%.pc: %.pc.in
	$(SED) -e 's|@PREFIX@|$(PREFIX)|g' -e 's|@VERSION@|$(VERSION)|g' > $@ < $<

install-header:
	$(MKDIR) -p $(PREFIX)/include/mufft
	$(INSTALL) -m644 fft.h $(PREFIX)/include/mufft

install-pkgconfig: $(PKGCONF_FILE)
	$(MKDIR) -p $(PREFIX)/lib/pkgconfig
	$(INSTALL) -m644 $(PKGCONF_FILE) $(PREFIX)/lib/pkgconfig

install-static: static install-pkgconfig install-header
	$(MKDIR) -p $(PREFIX)/lib
	$(MKDIR) -p $(PREFIX)/lib/pkgconfig
	$(INSTALL) -m644 $(TARGET_STATIC) $(PREFIX)/lib

install-shared: shared install-pkgconfig install-header
	$(MKDIR) -p $(PREFIX)/lib
	$(INSTALL) -m644 $(TARGET_SHARED) $(PREFIX)/lib

install-docs: docs
	$(MKDIR) -p $(PREFIX)/share/doc/mufft
	$(CP) -r docs/* $(PREFIX)/share/doc/mufft

install-nodocs: install-static install-shared

install: install-nodocs install-docs

.PHONY: all docs test shared static clean clean-all bench install install-header install-pkgconfig install-static install-shared install-docs install-nodocs

