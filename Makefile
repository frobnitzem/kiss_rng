# Alternative to cmake for quick testing.
example: example.cpp
	syclcc -I include -o example example.cpp
