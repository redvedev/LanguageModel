terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {}

# 1. Pobieranie obrazu Ubuntu 22.04
data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Oficjalne konto Canonical (twórców Ubuntu)

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]
  }
}

# 2. Rejestracja klucza SSH z Twojego komputera
resource "aws_key_pair" "deployer" {
  key_name   = "pytorch-key"
  public_key = file("~/.ssh/aws_pytorch_key.pub")
}

# 3. Reguły sieciowe - pozwalamy na łączenie się po SSH
resource "aws_security_group" "allow_ssh" {
  name        = "allow_ssh"
  description = "Allow SSH inbound traffic"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# 4. Główna maszyna EC2
resource "aws_instance" "pytorch_machine" {
  ami           = data.aws_ami.ubuntu.id
  instance_type = "t3.micro" # Darmowy tier
  key_name      = aws_key_pair.deployer.key_name
  vpc_security_group_ids = [aws_security_group.allow_ssh.id]

  # Ten skrypt wykonuje się raz, tuż po starcie maszyny
  user_data = <<-EOF
              #!/bin/bash
              # Dodanie 2GB pliku SWAP by PyTorch nie zabił maszyny brakiem RAMu
              fallocate -l 2G /swapfile
              chmod 600 /swapfile
              mkswap /swapfile
              swapon /swapfile
              echo '/swapfile none swap sw 0 0' | tee -a /etc/fstab

              # Aktualizacja i instalacja środowiska
              apt-get update
              apt-get install -y python3-pip python3-venv git

              # Instalacja lżejszej wersji PyTorch (tylko na procesor)
              pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
              EOF

  tags = {
    Name = "PyTorch-Free-Machine"
  }
}

# 5. Wyświetlenie adresu IP po stworzeniu maszyny
output "instance_public_ip" {
  description = "Publiczny adres IP Twojej maszyny"
  value       = aws_instance.pytorch_machine.public_ip
}
