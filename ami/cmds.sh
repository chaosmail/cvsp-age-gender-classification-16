# Configuration
INSTANCE=ec2-35-166-111-108.us-west-2.compute.amazonaws.com

# copy the CNN to the ami
scp -i ".ssh/us-west.pem" -rp cnn-keras "ec2-user@$INSTANCE"

# ssh to the instance
ssh -i ".ssh/us-west.pem" "ec2-user@$INSTANCE"

# Path to Dataset and S3 Bucket
cd ~/s3/imdb-wiki-dataset

