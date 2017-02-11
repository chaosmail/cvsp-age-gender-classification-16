# Update the installed packages
sudo yum update -y

sudo yum install -y htop hdparm

# Install Keras
sudo python3 -m pip install pip --upgrade
sudo python3 -m pip install keras pyfs sklearn
sudo python3 -m pip install jupyter --upgrade
sudo python3 -m pip install git+https://github.com/chaosmail/pyprind

# Add our public keys
echo "ssh-rsa <insert key>" >> /home/ec2-user/.ssh/authorized_keys

# Mount the shared SSD
sudo mkdir -p /data
sudo mount /dev/sdb /data
sudo chmod -R 777 /data