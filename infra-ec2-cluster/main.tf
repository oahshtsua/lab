# Backend Security Group
resource "aws_security_group" "backend_sg" {
  name        = "backend-sg"
  description = "Allow TCP inbound traffic on ${var.server_port} and all outbound traffic"

  tags = {
    Name = "backend-sg"
  }
}

resource "aws_vpc_security_group_ingress_rule" "allow_server_port_ipv4" {
  security_group_id = aws_security_group.backend_sg.id
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = var.server_port
  ip_protocol       = "tcp"
  to_port           = var.server_port
}

resource "aws_vpc_security_group_egress_rule" "allow_all_traffic_ipv4" {
  security_group_id = aws_security_group.backend_sg.id
  cidr_ipv4         = "0.0.0.0/0"
  ip_protocol       = "-1" # semantically equivalent to all ports
}

# Backend Launch Template
resource "aws_launch_template" "backend_lt" {
  name                   = "backend-lt"
  image_id               = "ami-0e35ddab05955cf57"
  instance_type          = var.instance_type
  vpc_security_group_ids = [aws_security_group.backend_sg.id]
  update_default_version = true

  user_data = base64encode(<<-EOF
    #!/bin/bash
    echo "<h1>Hello from $(hostname -f)" > index.html
    nohup busybox httpd -f -p ${var.server_port} &
    EOF
  )
}

# Backend Autoscaling Group
resource "aws_autoscaling_group" "backend_asg" {
  name             = "backend-asg"
  min_size         = var.servers_min
  max_size         = var.servers_max
  desired_capacity = var.servers_desired

  vpc_zone_identifier = data.aws_subnets.available.ids

  launch_template {
    id      = aws_launch_template.backend_lt.id
    version = "$Latest"
  }
}

# LoadBalancer Security Group
resource "aws_security_group" "backend_alb_sg" {
  name        = "backend-alb-sg"
  description = "Allow HTTP inbound traffic"

  tags = {
    Name = "backend-alb-sg"
  }
}

resource "aws_vpc_security_group_ingress_rule" "allow_http_traffic_ipv4" {
  security_group_id = aws_security_group.backend_alb_sg.id
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 80
  ip_protocol       = "tcp"
  to_port           = 80
}

resource "aws_vpc_security_group_egress_rule" "allow_server_port_ipv4" {
  security_group_id = aws_security_group.backend_alb_sg.id
  cidr_ipv4         = "0.0.0.0/0"
  from_port         = 8000
  ip_protocol       = "tcp"
  to_port           = 8000
}

# LoadBalancer Target Group
resource "aws_lb_target_group" "backend_alb_tg" {
  name     = "backend-alb-tg"
  port     = 8000
  protocol = "HTTP"
  vpc_id   = data.aws_vpc.default.id
}

# LoadBalancer
resource "aws_lb" "backend_alb" {
  name               = "backend-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.backend_alb_sg.id]
  subnets            = data.aws_subnets.available.ids
}

# LoadBalancer Listener Rule
resource "aws_lb_listener" "backend_alb_lr" {
  load_balancer_arn = aws_lb.backend_alb.arn
  port              = "80"
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.backend_alb_tg.arn
  }
}

# ALB Target Group Attachment
resource "aws_autoscaling_attachment" "backend_alb_asg" {
  autoscaling_group_name = aws_autoscaling_group.backend_asg.id
  lb_target_group_arn    = aws_lb_target_group.backend_alb_tg.arn
}
