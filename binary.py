from itertools import permutations


def zbits(n,k):
    """
       Generate all binary strings of length n that contain k zero bits
       
    Args:   
        n - the length of our desired output sequences.
        k - the number of zero-bits allowed in our output.
    Return:
        output - set of binary sequences
       
    """
    assert n > 0 and k > 0, 'invalid Data : n, k must be positive'
    assert int(n) == n and int(k) == k, 'invalid Data : n, k must be int'
    assert k <= n, 'invalid Data :k must be less than n'
    
    #initialize the binary string
    init_string = '0' * k + '1' * (n-k)
    
    #initialize the output set
    output = set()

    #generate all permutations of init_string and convert it into a string
    for item in permutations(init_string, n):
        output.add(''.join(item))
        
    return output

if __name__ == '__main__':
    assert zbits(4, 3) == {'0100', '0001', '0010', '1000'}
    assert zbits(4, 1) == {'0111', '1011', '1101', '1110'}
    assert zbits(5, 4) == {'00001', '00100', '01000', '10000', '00010'}
        
