# -*- coding: utf-8 -*-
"""
Created on Fri Jun  5 15:39:30 2015
Fortran Formatting Parser
Function to format Fortran nicely
@author: zag
"""

filename = 'C:\\Users\\zag\\Documents\\ArneSim\\QSpinor_src\\zachqspinor.f'
filename = 'C:\\Users\\zag\\Documents\\ArneSim\\QSpinor_src\\ChebyshevPropagator.f'

def read_in(filename):
    with open(filename, 'r') as fp:
        code = fp.readlines()
    return code
    
def readout(output):
    with open('parsed_output1.txt', 'w') as f:
        for i in output:
            ind = ''.join(' ' for k in range(4*i[1]))
            f.write(ind + i[0])


def filter_addition(code):
    """takes in a list of lines and appends so additions are on same line
    remove all whitespace from beginning of line"""
    #filter blanks and comments
    code = [i for i in code if not i.rstrip('\n').isspace()]
    code = [i for i in code if i != '\n']
    code = [i for i in code if i != '!     \n']
    code = [i for i in code if i[0] != '!']
    code = [i for i in code if i[0] != 'c']
    
    answer = []
    for i in range(len(code)):
        if code[i][5] == '+':
            answer[-1] = answer[-1].rstrip('\n') + code[i].lstrip('     +')
        else:
            answer.append(code[i])
    
    return [i.lstrip() for i in answer]
    
def filter_MPI(code):
    """filter for MPI statements"""
    ifmpi = [i for i in range(len(code)) if code[i][:6]=='#ifdef']
    endmpi = [i for i in range(len(code)) if code[i][:6] == '#endif']
    indices = [[ifmpi[i],endmpi[i]] for i in range(len(ifmpi))]
    good_ind = []
    for i in range(len(code)):
        print_it = True
        for j in indices:
            if i >= j[0] and i <= j[1]:
                print_it = False
        if print_it:  
            good_ind.append(i)
           
 
    return [code[i] for i in range(len(code)) if i in list(set(good_ind))]
        
def indent_code(filt):
    """assuming code is filltered as no indent list, keep track of indent and 
    write out.  create two dimensional list with each line and its indent level"""
    ind = 0
 
    answer = []
  
    for i in filt:
        if i[:2] == 'if' and 'then' in i:
            answer.append([i,ind])
            ind = ind + 1
           
        elif 'do' in i[:4]:
            answer.append([i,ind])
            ind = ind + 1
         
        elif 'endif' in i[:8] or 'end if' in i[:8]:
            ind = ind - 1
            answer.append([i,ind])
        elif i[:5] ==  'enddo':
            ind = ind - 1
            answer.append([i,ind])
        else:
            answer.append([i,ind])
    return answer
            

code = read_in(filename)
filt_code = filter_addition(code)
mpi_filt = filter_MPI(filt_code)
#ind_code = indent_code(filt_code)
ind_code = indent_code(mpi_filt)
readout(ind_code)
print('Done')
