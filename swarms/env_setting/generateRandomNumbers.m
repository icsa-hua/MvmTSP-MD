function numbers = generateRandomNumbers(nnumb, size) 
   
   if nnumb > size
        error('Number of numbers must be less than the size');
    end

    % Generate distinct random numbers
    numbers = randperm(size, nnumb);

end

