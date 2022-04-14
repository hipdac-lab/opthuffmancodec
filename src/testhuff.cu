/**
 * @file testhuff.cu
 * @author Cody Rivera
 * @brief
 * @version 0.0
 * @date 2022-04-13
 * (created) 2022-04-13

 * @copyright (C) 2022 by Washington State University, Argonne National Laboratory
 * See LICENSE in top-level directory
 *
 */

#include <iostream>
#include <fstream>
#include <limits>
#include <string>
#include <cstdlib>
#include <sys/stat.h>
#include "common.hh"
#include "utils.hh"
#include "huffman_coarse.cuh"
#include "huffman_parbook.cuh"

using UInt64 = unsigned long long;

size_t file_size(std::string filename) {
    struct stat stat_buf;
    int rc = stat(filename.c_str(), &stat_buf);
    return (size_t) (rc == 0 ? stat_buf.st_size : 0);
}

template <typename T, typename H, typename M = uint32_t>
void do_compress(string in_file, string out_file, int n_syms, size_t len) {
    using namespace cusz;
    int num_syms = std::min(n_syms, (int) std::numeric_limits<T>::max());

    Capsule<T> in_symbols(len, "Input symbols");
    Capsule<H> codebook(num_syms, "Codebook");
    Capsule<uint8_t> revbook(HuffmanCoarse<T, H>::get_revbook_nbyte(num_syms), "Reverse codebook");

    in_symbols.template alloc<cusz::LOC::HOST_DEVICE>();
    codebook.template alloc<cusz::LOC::DEVICE>();
    revbook.template alloc<cusz::LOC::HOST_DEVICE>();

    in_symbols.template from_file<cusz::LOC::HOST>(in_file)
        .host2device();

    HuffmanCoarse<T, H> codec;
    int sublen, pardeg;
    AutoconfigHelper::autotune(len, sublen, pardeg);
    codec.allocate_workspace(len, num_syms, pardeg);

    uint8_t* d_out;
    size_t out_len;
    codec.encode(
        in_symbols.template get<cusz::LOC::DEVICE>(),
        len,
        num_syms,
        sublen,
        pardeg,
        d_out,
        out_len
    );

    Capsule<uint8_t> out_codewords(out_len, "Output codewords");
    out_codewords.template set<cusz::LOC::DEVICE>(d_out)
        .template alloc<cusz::LOC::HOST>()
        .device2host();

    out_codewords.template to_file<cusz::LOC::HOST>(out_file + ".enc");

    in_symbols.template free<cusz::LOC::HOST_DEVICE>();
    codebook.template free<cusz::LOC::DEVICE>();
    revbook.template free<cusz::LOC::HOST_DEVICE>();
    out_codewords.template free<cusz::LOC::HOST_DEVICE>();
}

template <typename T, typename H, typename M = uint32_t>
void do_decompress(string in_file, string out_file, size_t len) {
    using namespace cusz;
    size_t in_len = file_size(in_file) / sizeof(uint8_t);

    Capsule<uint8_t> in_codewords(in_len, "Input codewords");
    Capsule<T> out_symbols(len, "Output symbols");

    in_codewords.template alloc<cusz::LOC::HOST_DEVICE>()
        .template from_file<cusz::LOC::HOST>(in_file)
        .host2device();

    out_symbols.template alloc<cusz::LOC::HOST_DEVICE>();

    HuffmanCoarse<T, H> codec;
    codec.decode(
        in_codewords.template get<cusz::LOC::DEVICE>(),
        out_symbols.template get<cusz::LOC::DEVICE>()
    );

    out_symbols.device2host();
    out_symbols.template to_file<cusz::LOC::HOST>(out_file + ".dec");

    in_codewords.template free<cusz::LOC::HOST_DEVICE>();
    out_symbols.template free<cusz::LOC::HOST_DEVICE>();
}


int main(int argc, char** argv) { 
    using namespace std;

    if (argc == 1) {
        cerr << "Usage: " << argv[0] << "(-z | -x) infile -l length [-o outfile] [--symtype (-s) u8|u16] [--cwtype (-c) u32|u64] [--nsyms (-n) num]" << endl;
        exit(0);
    }

    string in_file = "";
    string out_file = "";
    string sym_type = "u16", cw_type = "u32";
    string mode = "";
    int n_syms = 1024;
    size_t len = 0;

    int i = 1;
    while (i < argc) {
        if (argv[i][0] == '-') {
            auto long_opt = string(argv[i]);
            switch (argv[i][1]) {
                // ----------------------------------------------------------------
                case '-':
                    // string list
                    if (long_opt == "--symtype") goto sym_type;
                    if (long_opt == "--cwtype") goto cw_type;
                    if (long_opt == "--nsyms") goto n_syms;
                // ----------------------------------------------------------------
                case 'o':
                    if (i + 1 < argc) {
                        out_file = string(argv[i + 1]);
                        ++i;
                    }
                    break;
                case 's':
                sym_type:
                    if (i + 1 < argc) {
                        sym_type = string(argv[i + 1]);
                        ++i;
                    }
                    break;
                case 'c':
                cw_type:
                    if (i + 1 < argc) {
                        cw_type = string(argv[i + 1]);
                        ++i;
                    }
                    break;
                case 'n':
                n_syms:
                    if (i + 1 < argc) {
                        n_syms = stoi(string(argv[i + 1]));
                        ++i;
                    }
                    break;
                case 'l':
                    if (i + 1 < argc) {
                        len = (size_t) stoull(string(argv[i + 1]));
                        ++i;
                    }
                    break;
                case 'z':
                    mode = "compress";
                    break;
                case 'x':
                    mode = "decompress";
                    break;
                default:
                    cerr << "Bad option: " << argv[i] << endl;
                    exit(-1);
                    break;
            }
        }
        else {
            in_file = string(argv[i]);
        }
        ++i;
    }

    if (in_file == "") {
        cerr << "No input file, or no length specified" << endl;
        exit(-1);
    }

    if (out_file == "") {
        out_file = in_file;
    }

    if (mode == "compress") {
        if (len == 0) {
            cerr << "No length specified" << endl;
            exit(-1);
        }

        if (sym_type == "u8") {
            if (cw_type == "u32") {
                do_compress<uint8_t, uint32_t>(in_file, out_file, n_syms, len);
            } else if (cw_type == "u64") {
                do_compress<uint8_t, UInt64>(in_file, out_file, n_syms, len);
            }
        } else if (sym_type == "u16") {
            if (cw_type == "u32") {
                do_compress<uint16_t, uint32_t>(in_file, out_file, n_syms, len);
            } else if (cw_type == "u64") {
                do_compress<uint16_t, UInt64>(in_file, out_file, n_syms, len);
            }
        }
    } else {
        if (len == 0) {
            cerr << "No length specified" << endl;
            exit(-1);
        }

        if (sym_type == "u8") {
            if (cw_type == "u32") {
                do_decompress<uint8_t, uint32_t>(in_file, out_file, len);
            } else if (cw_type == "u64") {
                do_decompress<uint8_t, UInt64>(in_file, out_file, len);
            }
        } else if (sym_type == "u16") {
            if (cw_type == "u32") {
                do_decompress<uint16_t, uint32_t>(in_file, out_file, len);
            } else if (cw_type == "u64") {
                do_decompress<uint16_t, UInt64>(in_file, out_file, len);
            }
        }
    }

    return 0;
}
