from pydantic import BaseModel, EmailStr


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    message: str
    access_token: str


class RegisterRequest(BaseModel):
    username: str
    email: EmailStr
    password: str


class RegisterResponse(BaseModel):
    message: str
