git submodule update --init --recursive
cd acados
mkdir -p build && cd build
cmake -DACADOS_WITH_QPOASES=ON ..
make install -j4 && cd ..
sudo pip3 install interfaces/acados_template
make shared_library
