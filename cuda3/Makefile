NVCC = nvcc

as2: headers.h kernel.cu host.cu  
	$(NVCC) -I ./ kernel.cu host.cu -o as2

      
run:
	./as2

clean:
	rm -rf as2

