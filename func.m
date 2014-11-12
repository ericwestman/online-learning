function res = func(kernel, alpha, xi, x)
    res = 0;
    if numel(alpha) == 0
        return;
    end
    
    for i = 1:numel(alpha)
        res = res + alpha(i)*kernel(xi(i,:),x);
    end
    
end