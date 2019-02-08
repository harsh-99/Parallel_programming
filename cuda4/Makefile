NVCC = nvcc

as3: headers.h kernel.cu host.cu  
	$(NVCC) -I ./ kernel.cu host.cu -o as3

      
run:
	./as3

clean:
	rm -rf as3

