# Check the second EBS volume
sudo file -s /dev/xvdb
# > /dev/xvdb: data

# Create EXT4 filesystem on the partition
sudo mkfs -t ext4 /dev/xvdb

# Check the partition again
sudo file -s /dev/xvdb
# > /dev/xvdb: Linux rev 1.0 ext4 filesystem data, UUID=efb72105-1ac7-4877-920f-364a3b93f8bf (extents) (large files) (huge files)

# Mount the shared SSD
sudo mkdir -p /data
sudo mount /dev/sdb /data
sudo chmod -R 777 /data