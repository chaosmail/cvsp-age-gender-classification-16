# Update the installed packages
sudo yum update -y

# Install Keras
sudo python3 -m pip install keras pyfs

# S3 Credentials
sudo cat > /etc/passwd-s3fs <<EOF
chaosmail.dl.datastore:AKIAJFU62KYWFMP6VQFQ:l2qfsZwR1ZrgqKDZNSUUIYDm4xlqHgHTlISsFn9R
EOF
sudo chmod 600 /etc/passwd-s3fs

# Install S3FS
sudo sh setup_s3.sh

# Mount the shared SSD
sudo mkdir -p /data
sudo mount /dev/sdb /data
sudo chmod -R 777 /data

# Mount the S3 bucket
sudo mkdir -p /mnt/s3/datastore
sudo s3fs chaosmail.dl.datastore /mnt/s3/datastore -ouse_cache=/tmp