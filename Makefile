# Alternative to cmake for quick testing.
test: example/example.cpp
	syclcc -I include -o $@ $^
