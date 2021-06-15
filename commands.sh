aws s3api create-bucket --bucket ffs-codes --region eu-west-3 --create-bucket-configuration LocationConstraint=eu-west-3
aws s3api create-bucket --bucket ffs-images-gt --region eu-west-3 --create-bucket-configuration LocationConstraint=eu-west-3

aws cloudformation package --s3-bucket ffs-codes --template-file template.yaml --output-template-file generated/generated-ffs.yaml

aws cloudformation deploy --template-file generated/generated-ffs.yaml --stack-name ffs --capabilities CAPABILITY_IAM --region eu-west-3