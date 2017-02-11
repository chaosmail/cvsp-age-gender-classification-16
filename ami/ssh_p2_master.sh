USER="ec2-user"
PEM="~/.ssh/us-west.pem"
INSTANCE="ec2-35-165-247-246"
HOST="us-west-2.compute.amazonaws.com"

ssh -i "$PEM" $USER@$INSTANCE.$HOST

# As a user
ssh ec2-user@ec2-35-165-247-246.us-west-2.compute.amazonaws.com