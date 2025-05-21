variable "server_port" {
  description = "Port number on which the web server will listen for incoming HTTP requests"
  type        = number
}

variable "instance_type" {
  description = "The type of EC2 instance to launch"
  type        = string
}

variable "servers_min" {
  description = "Minimum number of server instances to maintain in the auto scaling group"
  type        = number
  default     = 1
}

variable "servers_max" {
  description = "Maximum number of server instances allowed in the auto scaling group"
  type        = number
  default     = 5
}

variable "servers_desired" {
  description = "Desired number of server instances to run in the auto scaling group under normal conditions"
  type        = number
  default     = 3
}

