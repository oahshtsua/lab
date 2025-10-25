package data

type User struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

type DataStore interface {
	GetUser(string) (*User, error)
	CreateUser(*User) error
}
