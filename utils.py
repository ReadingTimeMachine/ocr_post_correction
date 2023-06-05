import sys
import pandas as pd
import numpy as np
import altair as alt
import re
import fastwer

def error_and_quit(message,ignore_quit=False,warn=True):
    if warn: print(message)
    if not ignore_quit: sys.exit()

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


# for plotting, if you want to chop by tolerance
def subset_by_percent(dfin, tol = 0.01, verbose=True, round_off = 2, 
                     tol_count = None, reset_index = True, 
                     replace_insert = True, replace_deletion = True, 
                    track_insert_delete = False):
    """
    tol : in % (so 1.0 will be 1%, 0.1 will be 0.1%)
    tol_count : if not None, will over-write tol and subset by count
    """
    if tol_count is None:
        dfin_subset = dfin.loc[dfin['counts']> tol].copy()
    else:
        dfin_subset = dfin.loc[dfin['counts unnormalized']> tol_count].copy()

    # also, add the tool tip
    names = []
    for i in range(len(dfin_subset)):
        d = dfin_subset.iloc[i]
        names.append(str(round(d['counts'],2))+'%')
    dfin_subset['name']=names
    
    # rename columns for plotting 
    dfin_subset = dfin_subset.rename(columns={"counts": "% of all OCR tokens", 
                                              "counts unnormalized": "Total Count of PDF token"})
    if reset_index:
        dfin_subset = dfin_subset.reset_index(drop=True)
        
    # replace insert
    if replace_insert:
        dfin_subset.loc[(dfin_subset['ocr_letters']=='^')&(dfin_subset['pdf_letters']!='^'),'ocr_letters'] = 'INSERT'
    if replace_deletion:
        dfin_subset.loc[(dfin_subset['pdf_letters']=='@')&(dfin_subset['ocr_letters']!='@'),'pdf_letters'] = 'DELETE'
        
    d = dfin_subset.loc[(dfin_subset['ocr_letters']=='INSERT')&(dfin_subset['pdf_letters']=='DELETE')]
    if track_insert_delete:
        if len(d) > 0:
            print('Have overlap of insert and delete!')
            print(len(d))
    else: # assume error
        dfin_subset.loc[(dfin_subset['ocr_letters']=='INSERT')&(dfin_subset['pdf_letters']=='DELETE'),
                        '% of all OCR tokens'] = np.nan
        dfin_subset.loc[(dfin_subset['ocr_letters']=='INSERT')&(dfin_subset['pdf_letters']=='DELETE'),
                        "Total Count of PDF token"] = np.nan


    if verbose:
        print('shape of output=', dfin_subset.shape)
    return dfin_subset



# function for plots -- with histogramsmin_percent
def return_matrix_chart_withHist(dfin,  dfin_larger, textsize=20, stroke='black', 
                        height=800, width=900, scheme='viridis', 
                       log=True, color_title = 'Percent in %',
                       pdf_tag = 'GT', ocr_tag = 'OCR',
                       return_sort_ocr=False,
                       percent_column = "% of all OCR tokens",
                       count_column = "Total Count of PDF token",
                       pdf_title='PDF Characters', ocr_title='OCR Characters',
                                hist_width=800, min_percent = 1.0, hist_labelFontSize=16, 
                                hist_location = 'right', 
                            legend_length = 200, legend_Y = -50,legend_direction='horizontal',
                                color_selection = False,
                                insert_delete_at_end = True, 
                                labelFontSize=20,titleFontSize=20,
                                plot_hist = True):
    
    # for colormap legend
    # legend placement
    length = legend_length
    legendY = legend_Y
    legendX = 0 + width//2 - length//2

    
    sort_pdf = np.unique(dfin['pdf_letters']).tolist()
    sort_pdf.sort()
    extra_ocr = []
    for o in dfin['ocr_letters'].unique():
        if o not in sort_pdf:
            extra_ocr.append(o)

    extra_ocr.sort()

    sort_ocr = sort_pdf.copy()
    sort_ocr.extend(extra_ocr)
    
    # move both delete and insert to the end(?)
    if insert_delete_at_end:
        if 'INSERT' in sort_ocr:
            i = sort_ocr.index('INSERT')
            sort_ocr.pop(i)
            sort_ocr.append('INSERT')
        if 'DELETE' in sort_pdf:
            i = sort_pdf.index('DELETE')
            sort_pdf.pop(i)
            sort_pdf.append('DELETE')
    
    # check special characters:
    for i in range(len(sort_pdf)):
        if '\\' in repr(sort_pdf[i]): # have escaped
            sort_pdf[i] = re.escape(repr(sort_pdf[i]))
        
    for i in range(len(sort_ocr)):
        if '\\' in repr(sort_ocr[i]): # have escaped
            sort_ocr[i] = re.escape(repr(sort_ocr[i]))
            
    # also clean dataframe
    for i in range(len(dfin)):
        for c in ['pdf_letters','ocr_letters']:
            esc = repr(dfin.iloc[i][c])
            if '\\' in esc: # have escaped
                dfin.at[i,c]=re.escape(esc)
                    
    # maybe some special words?
    for iss,s in enumerate(sort_ocr):
        if 'if' == s:
            #sort_ocr[iss] = "''if"
            sort_ocr[iss] = '"if"' #str('if')
            #sort_ocr[iss] = u'if'
    # maybe some special words?
    for iss,s in enumerate(sort_pdf):
        if 'if' == s:
            #sort_pdf[iss] = "''if"
            sort_pdf[iss] = '"if"' #str('if')
            #sort_pdf[iss] = u'if'


    if color_selection:
        column_select = alt.selection_point(fields=['column'],
                                     bind=alt.binding_select(options=[percent_column, 
                                                                      count_column], 
                                                             name='Color by: '),
                                     value=percent_column)
        color_col = 'value'
    else:
        color_col = percent_column

    selector = alt.selection_point(encodings=['y'])#, init={pdf_letters:'A'})
    opacity = alt.condition(selector,alt.value(1),alt.value(0.25))

    if log:
        if not color_selection:
            color = alt.Color(color_col+":Q", 
                              scale=alt.Scale(type='log',scheme=scheme,
                                              domain=[min_percent,100]),
                              title=color_title,
                              legend=alt.Legend(
                            orient='none',
                            legendX=legendX, legendY=legendY,
                            direction=legend_direction,
                            titleAnchor='middle', gradientLength=length))
        else:
            color = alt.Color(color_col+":Q", 
                              scale=alt.Scale(type='log',
                                              scheme=scheme),
                              title=color_title,
                              legend=alt.Legend(
                            orient='none',
                            legendX=legendX, legendY=legendY,
                            direction=legend_direction,
                            titleAnchor='middle', gradientLength=length))
    else:
        if not color_selection:
            color = alt.Color(color_col+":Q", 
                              scale=alt.Scale(scheme=scheme,
                                              domain=[min_percent,100]),
                              title=color_title,
                              legend=alt.Legend(
                            orient='none',
                            legendX=legendX, legendY=legendY,
                            direction=legend_direction,
                            titleAnchor='middle', gradientLength=length))
        else:
            color = alt.Color(color_col+":Q", scale=alt.Scale(scheme=scheme),
                              title=color_title,
                              legend=alt.Legend(
                            orient='none',
                            legendX=legendX, legendY=legendY,
                            direction=legend_direction,
                            titleAnchor='middle', gradientLength=length))
        
    if not color_selection:
        chart1 = alt.Chart(dfin).mark_rect().transform_fold(
            fold=[percent_column, count_column],
            as_=['column', 'value']
        ).encode(
            alt.Y("pdf_letters:O",sort=sort_pdf,title=pdf_title),
            alt.X("ocr_letters:O",sort=sort_ocr,title=ocr_title),
            color=color,
            opacity=opacity,
            tooltip=[alt.Tooltip("pdf_letters:O",title=pdf_tag), 
                     alt.Tooltip("ocr_letters:O",title=ocr_tag), 
                     alt.Tooltip("name:N",title='Percentage'),
                    alt.Tooltip(count_column+':Q',title='Count')]
        ).properties(
            height=height,
            width=width
        ).add_params(
            selector
        )

    else:
        chart1 = alt.Chart(dfin).mark_rect().transform_fold(
            fold=[percent_column, count_column],
            as_=['column', 'value']
        ).transform_filter(
            column_select
        ).encode(
            alt.Y("pdf_letters:O",sort=sort_pdf,title=pdf_title),
            alt.X("ocr_letters:O",sort=sort_ocr,title=ocr_title),
            color=color,
            opacity=opacity,
            tooltip=[alt.Tooltip("pdf_letters:O",title=pdf_tag), 
                     alt.Tooltip("ocr_letters:O",title=ocr_tag), 
                     alt.Tooltip("name:N",title='Percentage'),
                    alt.Tooltip(count_column+':Q',title='Count')]
        ).properties(
            height=height,
            width=width
        ).add_params(
            selector,
            column_select
        )
        
    if plot_hist:
        chart2 = alt.Chart(dfin_larger).mark_bar().transform_filter(
            selector
        ).transform_filter(
           alt.FieldRangePredicate(field=percent_column, range=[100, min_percent])
           #alt.FieldRangePredicate(field=percent_column, range=[100, slider])
        ).encode(
            alt.X('ocr_letters:O', sort='-y',title=ocr_title),#,labelFontSize=hist_labelFontSize),
            alt.Y("% of all OCR tokens:Q"),#, 
                tooltip=[alt.Tooltip("pdf_letters:O",title=pdf_tag), 
                     alt.Tooltip("ocr_letters:O",title=ocr_tag), 
                     alt.Tooltip("name:N",title='Percentage'),
                    alt.Tooltip(count_column+':Q',title='Count')]

        ).properties(
            width=hist_width
        )

        if hist_location == 'bottom':
            chart = alt.vconcat(chart1, chart2, center=True).configure_axis(
                labelFontSize=labelFontSize,
                titleFontSize=titleFontSize
            )
        elif hist_location == 'right':
            chart = alt.hconcat(chart1,chart2,center=True).configure_axis(
                labelFontSize=labelFontSize,
                titleFontSize=titleFontSize
            )
        else:
            print('not supported location for hist, will place on right')
            chart = alt.hconcat(chart1,chart2,center=True).configure_axis(
                labelFontSize=labelFontSize,
                titleFontSize=titleFontSize
            )
    else:
        chart = alt.vconcat(chart1, center=True).configure_axis(
                labelFontSize=labelFontSize,
                titleFontSize=titleFontSize
            )
        

    if return_sort_ocr:
        return chart, sort_ocr
    #return chart,chart1    
    return chart


def return_dropdown_hist(dfin_larger,
    percent_column = "% of all OCR tokens",
    count_column = "Total Count of PDF token",
    pdf_tag = 'GT',
    ocr_tag = 'OCR',
    hist_width=800,
    min_percent = 0.01,
    pdf_title='PDF Characters',
    ocr_title='OCR Characters',
                        ylog=False):

    input_dropdown = alt.binding_select(options=np.sort(dfin_larger['pdf_letters'].unique()), 
                                        name=pdf_title + ': ')
    selector = alt.selection_point(fields=['pdf_letters'], 
                                   bind=input_dropdown,
                            value=np.sort(dfin_larger['pdf_letters'].unique())[0])

    yaxis = alt.Y("% of all OCR tokens:Q",
                 scale=alt.Scale(domain=[min_percent,100]))
    if ylog:
        yaxis = alt.Y("% of all OCR tokens:Q", 
                      scale=alt.Scale(type='symlog',
                                      domain=[min_percent,100]))
        
    chart2 = alt.Chart(dfin_larger).mark_bar().transform_filter(
        selector
    ).transform_filter(
       alt.FieldRangePredicate(field=percent_column, range=[100, min_percent])
    ).encode(
        alt.X('ocr_letters:O', sort='-y',title=ocr_title),
        yaxis,
            tooltip=[alt.Tooltip("pdf_letters:O",title=pdf_tag), 
                 alt.Tooltip("ocr_letters:O",title=ocr_tag), 
                 alt.Tooltip("name:N",title='Percentage'),
                alt.Tooltip(count_column+':Q',title='Count')]
    ).add_params(
        selector
    ).properties(
        width=hist_width
    )
    
    return chart2


# to "fix" OCR (found and GT) for specific markers
def fix_ocr(dfout_historical, 
            marks={'citations':'↫','refs':'↷','inlines':'↭'}, 
            predicted_text_col = 'predicted_text',
           search_doc = r"\\(?:[^a-zA-Z]|[a-zA-Z]+[*=']?)"):
    dfout_err_h = dfout_historical.copy()

    dfmask = []

    cer_orig = []; wer_orig = []
    cer_corr = []; wer_corr = []
    cer_corr_fix = []; wer_corr_fix = []
    cer_fix = []; wer_fix = []
    pdf_fix_out = []; ocr_fix_out = []
    for i in range(len(dfout_historical)):

        ocr = dfout_historical.iloc[i]['input_text']
        pdf = dfout_historical.iloc[i]['target_text']
        ocr_corr = dfout_historical.iloc[i][predicted_text_col]

 
        if pd.isnull(ocr) or pd.isnull(pdf) or pd.isnull(ocr_corr):
            pdf_fix_out.append(np.nan)
            ocr_fix_out.append(np.nan)
            cer_orig.append(np.nan)
            wer_orig.append(np.nan)
            cer_corr.append(np.nan)
            wer_corr.append(np.nan)
            cer_corr_fix.append(np.nan)
            wer_corr_fix.append(np.nan)
            cer_fix.append(np.nan)
            wer_fix.append(np.nan)
            dfmask.append(False)
            continue
            
        
       # orig
        ocr_orig = str(dfout_historical.iloc[i]['input_text'])
        pdf_orig = str(dfout_historical.iloc[i]['target_text'])
        ocr_corr_orig = str(dfout_historical.iloc[i]['predicted_text'])
        
        # count '$'
        #pdf_orig = str(pdf_orig)
        if list(pdf_orig).count('$')%2 != 0:
            pdf_fix_out.append(np.nan)
            ocr_fix_out.append(np.nan)
            cer_orig.append(np.nan)
            wer_orig.append(np.nan)
            cer_corr.append(np.nan)
            wer_corr.append(np.nan)
            cer_corr_fix.append(np.nan)
            wer_corr_fix.append(np.nan)
            cer_fix.append(np.nan)
            wer_fix.append(np.nan)
            dfmask.append(False)
            continue
            # print('uneven $!')
            # print(pdf_orig)
            # import sys; sys.exit()


            
        # if not, format
        ocr = str(ocr); pdf = str(pdf)
        ocr_corr = str(ocr_corr)

        # also, do with just taking out locations of citations, refs and inlines:
        # ------ 1. inline math --------
        ind = 0
        pdf_fix = ''
        while ind < len(pdf):
            if '$' in pdf[ind:]:
                i1 = pdf[ind:].index('$')
                try:
                    i2 = pdf[ind+i1+1:].index('$')+i1+1+1
                    # find match
                    pdf_fix += pdf[ind:ind+i1]
                    pdf_fix += marks['inlines']
                    #import sys; sys.exit()
                    ind += i2
                except:
                    print('matching $ not found, moving on...')
                    print('current:', pdf)
                    print('prior:', pdf_orig)
                    import sys; sys.exit()
                    ind += i1
            else:
                pdf_fix += pdf[ind:]
                ind += len(pdf[ind:])

        ind = 0
        ocr_corr_fix = ''
        while ind < len(ocr_corr):
            if '$' in ocr_corr[ind:]:
                i1 = ocr_corr[ind:].index('$')
                nope = False
                try:
                    i2 = ocr_corr[ind+i1+1:].index('$')+i1+1+1
                except:
                    print('no matching $ in OCR corrected fix!')
                    nope = True
                if not nope:
                    # find match
                    ocr_corr_fix += ocr_corr[ind:ind+i1]
                    ocr_corr_fix += marks['inlines']
                    #import sys; sys.exit()
                    ind += i2
                else:
                    ind += i1+1
            else:
                ocr_corr_fix += ocr_corr[ind:]
                ind += len(ocr_corr[ind:])

        # ------ 2. citations -----
        ind = 0
        pdf_fix2 = ''
        while ind < len(pdf_fix):
            if '\\cite' in pdf_fix[ind:]:
                ind1,ind2,err = split_function_with_delimiters_with_checks(pdf_fix[ind:],function='\\cite',
                        dopen='{',dclose='}',
                       error_out=False)
                if ind1 != -1 and ind2 != -2:
                    pdf_fix2 += pdf_fix[ind:ind+ind1]
                    pdf_fix2 += marks['citations']
                    ind += ind2
                else:
                    ind += pdf_fix[ind:].index('\\cite')+len('\\cite')
            else:
                pdf_fix2 += pdf_fix[ind:]
                ind += len(pdf_fix[ind:])

        ind = 0
        ocr_corr_fix2 = ''
        while ind < len(ocr_corr_fix):
            if '\\cite' in ocr_corr_fix[ind:]:
                ind1,ind2,err = split_function_with_delimiters_with_checks(ocr_corr_fix[ind:],function='\\cite',
                        dopen='{',dclose='}',
                       error_out=False)
                if not err:
                    ocr_corr_fix2 += ocr_corr_fix[ind:ind+ind1]
                    ocr_corr_fix2 += marks['citations']
                    ind += ind2
                else:
                    ind += ocr_corr_fix[ind:].index('\\cite')+len('\\cite')
            else:
                ocr_corr_fix2 += ocr_corr_fix[ind:]
                ind += len(ocr_corr_fix[ind:])

         # ------ 3. refs -----
        ind = 0
        pdf_fix3 = ''
        while ind < len(pdf_fix2):
            if '\\ref' in pdf_fix2[ind:]:
                ind1,ind2 = split_function_with_delimiters_with_checks(pdf_fix2[ind:],function='\\ref',
                        dopen='{',dclose='}',
                       error_out=True)
                pdf_fix3 += pdf_fix2[ind:ind+ind1]
                pdf_fix3 += marks['refs']
                ind += ind2
            else:
                pdf_fix3 += pdf_fix2[ind:]
                ind += len(pdf_fix2[ind:])

        ind = 0
        ocr_corr_fix3 = ''
        while ind < len(ocr_corr_fix2):
            if '\\ref' in ocr_corr_fix2[ind:]:
                ind1,ind2 = split_function_with_delimiters_with_checks(ocr_corr_fix2[ind:],function='\\ref',
                        dopen='{',dclose='}',
                       error_out=True)
                ocr_corr_fix3 += ocr_corr_fix2[ind:ind+ind1]
                ocr_corr_fix3 += marks['refs']
                ind += ind2
            else:
                ocr_corr_fix3 += ocr_corr_fix2[ind:]
                ind += len(ocr_corr_fix2[ind:])

        #if i == 5: import sys; sys.exit()

        # orig errors
        cer_orig_here = fastwer.score_sent(ocr_orig,pdf_orig, 
                                      char_level=True)
        wer_orig_here = fastwer.score_sent(ocr_orig,pdf_orig, 
                                      char_level=False)
        # after correction
        cer_corr_here = fastwer.score_sent(ocr_corr_orig,pdf_orig, 
                                      char_level=True)
        wer_corr_here = fastwer.score_sent(ocr_corr_orig,pdf_orig, 
                                      char_level=False)

        # after correction, with replacement
        cer_corr_fix_here = fastwer.score_sent(ocr_corr_fix3,pdf_fix3, 
                                      char_level=True)
        wer_corr_fix_here = fastwer.score_sent(ocr_corr_fix3,pdf_fix3, 
                                      char_level=False)

        # before correction, with replacement
        cer_fix_here = fastwer.score_sent(ocr,pdf_fix3, 
                                      char_level=True)
        wer_fix_here = fastwer.score_sent(ocr,pdf_fix3, 
                                      char_level=False)


        # ignore if we haven't checked for extra non-tracked things
        pdf_check = pdf_fix3.replace('*','').replace('`','').replace('$','')
        s = re.search(search_doc, pdf_check)
        if s: # skip
            pdf_fix_out.append(np.nan)
            ocr_fix_out.append(np.nan)
            cer_orig.append(np.nan)
            wer_orig.append(np.nan)
            cer_corr.append(np.nan)
            wer_corr.append(np.nan)
            cer_corr_fix.append(np.nan)
            wer_corr_fix.append(np.nan)
            cer_fix.append(np.nan)
            wer_fix.append(np.nan)
            dfmask.append(False)
            continue
        else:
            dfmask.append(True)


        pdf_fix_out.append(pdf_fix3)
        ocr_fix_out.append(ocr_corr_fix3)


        cer_orig.append(cer_orig_here)
        wer_orig.append(wer_orig_here)
        cer_corr.append(cer_corr_here)
        wer_corr.append(wer_corr_here)
        cer_corr_fix.append(cer_corr_fix_here)
        wer_corr_fix.append(wer_corr_fix_here)
        cer_fix.append(cer_fix_here)
        wer_fix.append(wer_fix_here)

    # recopy
    #dfout_err_h = dfout_historical.copy().loc[dfmask]

    dfout_err_h['CER Orig'] = cer_orig
    dfout_err_h['WER Orig'] = wer_orig
    dfout_err_h['CER Corrected'] = cer_corr
    dfout_err_h['WER Corrected'] = wer_corr
    dfout_err_h['CER Fix Corrected'] = cer_corr_fix
    dfout_err_h['WER Fix Corrected'] = wer_corr_fix
    dfout_err_h['CER Fix'] = cer_fix
    dfout_err_h['WER Fix'] = wer_fix
    dfout_err_h['target_text_fixed'] = pdf_fix_out
    dfout_err_h['predicted_text_fixed'] = ocr_fix_out
    dfout_err_h['masked entries'] = dfmask
    
    return dfout_err_h

# input a word-type, output the alignment character
def select_indicator(wordtype):
    if wordtype == 'word':
        indicator = {'pdf': 'W', 'ocr':'W'}
    elif wordtype == 'inline':
        indicator = {'pdf': 'I', 'ocr':'I'}
    elif wordtype == 'cite':
        indicator = {'pdf': 'C', 'ocr':'C'}
    elif wordtype == 'hyp-cite-top':
        indicator = {'pdf': 'c', 'ocr-top':'d', 'ocr-bottom':'e'}
    elif wordtype == 'hyp-word-top':
        indicator = {'pdf': 'w', 'ocr-top':'x', 'ocr-bottom':'y'}
    elif wordtype == 'ref':
        indicator = {'pdf': 'R', 'ocr':'R'}
    elif wordtype == 'hyp-inline-top':
        indicator = {'pdf': 'i', 'ocr-top':'j', 'ocr-bottom':'k'}
    elif wordtype == 'hyp-ref-top':
        indicator = {'pdf': 'r', 'ocr-top':'q', 'ocr-bottom':'s'}
    else:
        print('no idea!!!')
        import sys; sys.exit()
    return indicator       

# reverse -- input character, output the type of character
def select_wordtype(c):
    if c == 'W':
        t = 'word' # does not depend on char_type
    elif c =='I':
        t = 'inline'
    elif c == 'C':
        t = 'cite'
    elif c == 'c':
        t = 'hyp-cite' # only SGT
    elif c == 'd':
        t = 'hyp-cite-top' # only OCR
    elif c == 'e':
        t = 'hyp-cite-bottom' # only OCR
    elif c == 'R':
        t = 'ref'
    elif c == 'w':
        t = 'hyp-word' # only SGT
    elif c == 'x':
        t = 'hyp-word-top' # only OCR
    elif c == 'y':
        t = 'hyp-word-bottom' # only OCR
    elif c == 'i':
        t = 'hyp-inline' # only SGT
    elif c == 'j':
        t = 'hyp-inline-top' # only OCR
    elif c == 'k':
        t = 'hyp-inline-bottom' # only OCR
    elif c == 'r':
        t = 'hyp-ref' # only SGT
    elif c == 'q':
        t = 'hyp-ref-top' # only OCR
    elif c == 's':
        t = 'hyp-ref-bottom' # only OCR
    elif c == '^':
        t = 'insert' # only OCR
    elif c == '@': # only PDF
        t = 'delete'
    elif c == ' ': # just a space!
        t = ' '
    else:
        print('no idea here!')
        import sys; sys.exit()
    return t
    