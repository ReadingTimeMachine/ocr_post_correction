#https://stackoverflow.com/questions/29991917/indices-of-matching-parentheses-in-python
def find_closing(text,dopen='(',dclose=')', debug=True,
                 remove_newline=False, check_closing=True):
    istart = []  # stack of indices of opening parentheses
    d = {}
    error = False
    #print('------text------')
    #print(text)
    #print('----------------')
    if remove_newline: text = text.strip('\n')
    text = text + '     '
    for i, c in enumerate(text):
        if c == dopen:
            istart.append(i)
        if c == dclose:
            try:
                d[istart.pop()] = i
            except IndexError:
                if debug: print('Too many closing parentheses')
                error=True
    if len(istart)!=0:  # check if stack is empty afterwards or left over openers?
        if debug: print('Too many opening parentheses, check 2')
        error=True
    if error and not check_closing: error = False
    return d, error

def split_function_with_delimiters(l,function='\\footnote',
                                   dopen='{',dclose='}',
                                   debug=False, 
                                  remove_newline=False,
                                  check_closing=True,
                                  start_after_function=False,
                                  verbose=False,
                                  return_bracket_index=False):
    if function not in l: # no function
        if debug: print('function not in text:', function)
        return -1, -1
    ind1 = l.index(function)
    if start_after_function:
        ind1 = ind1+len(function)
    tt = l[ind1:]
    # find first bracket
    if dopen not in l[ind1:]: # no first bracket
        if debug: print('no opening bracket in text')
        return -1,-1
    ind2 = l[ind1:].index(dopen)
    tt2 = l[ind1+ind2:]
    #print('here')
    #debug=True
    d,error = find_closing(tt2,dopen=dopen,dclose=dclose,
                           debug=debug,
                           remove_newline=remove_newline,
                          check_closing=check_closing)
    #if error:
    #    import sys; sys.exit()
    if error:
        if debug: print('error in find_closing')
        return ind1,-1
    #print('also here')
    try:
        ind3 = ind1+ind2+d[0]+1
    except:
        ind3 = -1; ind1 = -1
        if verbose: print(d)
        #error_and_quit('oop!')
        if verbose: print('in split_function_with_delimiters: no d[0]')
    #l[ind1:ind3]
    if start_after_function: ind1 = ind1-len(function)
    if not return_bracket_index:
        return ind1,ind3 # ind1 starts at the start of the function, ind3 at its closing bracket
    else:
        return ind2,ind3 # ind2 start of braket, ind3 is end of bracket


def split_function_with_delimiters_with_checks(l,function='\\footnote',
                                   dopen='{',dclose='}', error_tag='',
                                              start_after_function=False,
                                              error_out = True,
                                              return_bracket_index=False,
                                              verbose=False):
    error=False
    ind1,ind2 = split_function_with_delimiters(l,
                                                   function=function,
                                                   dopen=dopen,
                                                   dclose=dclose,
                                                   debug=False,
                                                  check_closing=False,
                                              start_after_function=start_after_function,
                                              return_bracket_index=return_bracket_index,
                                              verbose=verbose)
    if ind1 == -1 or ind2 == -1: # re-run w/o check closing
        if verbose: print(error_tag+' :not found, trying again w/o checking closing...')
        ind1,ind2 = split_function_with_delimiters(l,
                                                   function=function,
                                                   dopen=dopen,
                                                   dclose=dclose,
                                                   debug=False,
                                                  check_closing=False,
                                                  start_after_function=start_after_function,
                                                  return_bracket_index=return_bracket_index,
                                                  verbose=verbose)
        if ind1 == -1 or ind2 == -1: # re-run w/o check closing
            ind1,ind2 = split_function_with_delimiters(l,
                                                       function=function,
                                                       dopen=dopen,
                                                       dclose=dclose,
                                                       debug=verbose,
                                                      check_closing=False,
                                                      start_after_function=start_after_function,
                                                      return_bracket_index=return_bracket_index,
                                                      verbose=verbose)
            if error_out:
                print(function)
                print('---------')
                print(l)
                error_and_quit(error_tag+' : couldnt figure it out!')
            else:
                if verbose: print(error_tag+' : couldnt figure it out!')
                error=True
    if error_out:
        return ind1,ind2 # ind1 starts at the start of the function, ind3 at its closing bracket
    else:
        return ind1,ind2,error

    
def align_texts_fast(ocr_text, pdf_text, eops, 
                     pdf_types = None, ocr_types = None):
    
    pdf_text_aligned2 = list(pdf_text)
    ocr_text_aligned2 = list(pdf_text)

    use_types = False

    nadd = 0
    if pdf_types is not None and ocr_types is not None:
        pdf_type_aligned2 = list(pdf_types)
        ocr_type_aligned2 = list(pdf_types)
        use_types = True

    for i in range(len(eops)):
        # op, source index (ocr), destination index (pdf)
        operation, sp,dp = eops[i]
        if operation == 'replace': # there is a replacement
            ocr_text_aligned2[dp+nadd] = ocr_text[sp]
            if use_types:
                ocr_type_aligned2[dp+nadd] = ocr_types[sp]

        elif operation == 'insert': # insert in source (OCR)
            ocr_text_aligned2[dp+nadd] = '^'
            if use_types:
                ocr_type_aligned2[dp+nadd] = '^'

        elif operation == 'delete': # take off in OCR --> same as insert in PDF
            pdf_text_aligned2.insert(dp+nadd,'@')
            ocr_text_aligned2.insert(dp+nadd, ocr_text[sp])
            if use_types:
                ocr_type_aligned2.insert(dp+nadd, ocr_types[sp])
                pdf_type_aligned2.insert(dp+nadd, '@')
            nadd += 1 # update because we have now made the destination string longer
        else:
            print('unknown operation')
            import sys; sys.exit()

    pdf_text_aligned2 = "".join(pdf_text_aligned2)
    ocr_text_aligned2 = "".join(ocr_text_aligned2)

    if use_types:
        pdf_type_aligned2 = "".join(pdf_type_aligned2)
        ocr_type_aligned2 = "".join(ocr_type_aligned2)
                      

#     pdf_text_aligned2 = list(pdf_text)
#     ocr_text_aligned2 = list(pdf_text)
    
#     use_types = False
#     if pdf_types is not None and ocr_types is not None:
#         pdf_type_aligned2 = list(pdf_types)
#         ocr_type_aligned2 = list(pdf_types)
#         use_types = True
        

#     for i in range(len(eops)):
#         # op, source index (ocr), destination index (pdf)
#         operation, sp,dp = eops[i]
#         if operation == 'replace': # there is a replacement
#             #pass
#             ocr_text_aligned2[dp] = ocr_text[sp]
#             if use_types:
#                 ocr_type_aligned2[dp] = ocr_types[sp]
                
#         elif operation == 'insert': # insert in source (OCR)
#             #ocr_text_aligned2.insert(dp,'^')
#             ocr_text_aligned2[dp] = '^'
#             if use_types:
#                 ocr_type_aligned2[dp] = '^'
#         elif operation == 'delete': # take off in OCR --> same as insert in PDF
#             pdf_text_aligned2.insert(dp,'@')
#             ocr_text_aligned2.insert(dp, ocr_text[sp])
#             if use_types:
#                 ocr_type_aligned2.insert(dp, ocr_types[sp])
#                 pdf_type_aligned2.insert(dp, '@')
#         else:
#             print('unknown operation')
#             import sys; sys.exit()

#     pdf_text_aligned2 = "".join(pdf_text_aligned2)
#     ocr_text_aligned2 = "".join(ocr_text_aligned2)
    
#     if use_types:
#         pdf_type_aligned2 = "".join(pdf_type_aligned2)
#         ocr_type_aligned2 = "".join(ocr_type_aligned2)
        

    if not use_types:
        return ocr_text_aligned2, pdf_text_aligned2
    else:
        return ocr_text_aligned2, pdf_text_aligned2, ocr_type_aligned2, pdf_type_aligned2
        

# fill in types
def get_fill_in_types(pdf_text_aligned_all_types):
    # fill between in PDF @'s
    fill_in_types = list(pdf_text_aligned_all_types)
    for ifill in range(len(fill_in_types)):
        if ifill==0:
            if fill_in_types[ifill] == '@' and fill_in_types[ifill+1] != ' ':
                fill_in_types[ifill] = fill_in_types[ifill+1]
        else:
            if fill_in_types[ifill] == '@':
                if fill_in_types[ifill-1] != ' ':
                    fill_in_types[ifill] = fill_in_types[ifill-1]
    fill_in_types = "".join(fill_in_types) 
    #realign_pages[pk]['PDF full fill types'] = fill_in_types
    return fill_in_types
