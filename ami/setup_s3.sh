# Installing S3FS
# https://github.com/s3fs-fuse/s3fs-fuse/wiki/Installation-Notes#installing-on-amazon-linux-ami


# Install some build tools
yum install -y gcc libstdc++-devel gcc-c++ fuse fuse-devel curl-devel libxml2-devel mailcap automake openssl-devel

git clone https://github.com/s3fs-fuse/s3fs-fuse /opt/s3fs-fuse

cd /opt/s3fs-fuse

./autogen.sh
./configure --prefix=/usr --with-openssl
make
make install

cd