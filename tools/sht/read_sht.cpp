// Decode with the upstream SHTfile reader, including its checksum check.
#include <algorithm>
#include <cstring>
#include <fstream>
#include <limits>
#include "sht_file.hpp"

int main(int argc, char** argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: emsoft-di-read-sht INPUT.sht");
        std::ifstream input(argv[1], std::ios::binary);
        sht::File file;
        file.read(input);
        if (file.header.modality() != sht::Modality::EBSD)
            throw std::runtime_error("expected an EBSD master");
        const int bw = file.harmonics.bw();
        if (bw < 1 || bw > 1024) throw std::runtime_error("unsupported SHT bandwidth");
        const auto& h = file.harmonics;
        if (h.alm.size() != sht::HarmonicsData::NumHarm(bw, h.zRot(), h.cmpFlg()))
            throw std::runtime_error("incorrect packed coefficient count");
        std::vector<std::complex<double>> coefficients(size_t(bw) * bw);
        file.harmonics.unpackHarm(h.alm.data(), coefficients.data());
        std::cout << bw << ' ' << int(file.mpData.sgEff()) << ' '
                  << file.header.beamEnergy() << '\n';
        std::cout.write(reinterpret_cast<const char*>(coefficients.data()),
                        coefficients.size() * sizeof(coefficients[0]));
        return std::cout ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
