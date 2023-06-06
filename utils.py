import sys
import pandas as pd
import numpy as np
import altair as alt
import re
import fastwer
from pylatexenc.latex2text import LatexNodes2Text

# for counting pdf pages
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfinterp import resolve1

from wand.image import Image as WandImage
from wand.color import Color
from PIL import Image

import cv2 as cv
import numpy as np
import os

from lxml import etree
from PIL import Image, ImageDraw, ImageFont


sp = 65536. * 72.27/72, # latex and printers are different
sp_printer = sp # naming stuff, this is bad code


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
    
    
# for when passing a list with comments already taken out
def get_tikz_preamble():

    error = False
    t2 = [] # store tikzmark lines
    t2.append('\\usepackage{tikz}\n\\usetikzlibrary{tikzmark}\n')
    # libraries 
    t2.append('\\usepackage{atbegshi,ifthen,listings}')
    t2.append('\\tikzstyle{highlighter} = [yellow,line width = \\baselineskip,]')
    t2.append('% from: https://tex.stackexchange.com/questions/15237/highlight-text-in-code-listing-while-also-keeping-syntax-highlighting/49309#49309')
    t2.append('\\newcounter{highlight}[page]')
    t2.append('\\newcommand{\\tikzhighlightanchor}[1]{\\ensuremath{\\vcenter{\\hbox{\\tikz[remember picture, overlay]{\\coordinate (#1 highlight \\arabic{highlight});}}}}}')
    t2.append('\\newcommand{\\bh}[0]{\\stepcounter{highlight}\\tikzhighlightanchor{begin}}')
    t2.append('\\newcommand{\\eh}[0]{\\tikzhighlightanchor{end}}')
    t2.append('\\AtBeginShipout{\\AtBeginShipoutUpperLeft{\\ifthenelse{\\value{highlight} > 0}{\\tikz[remember picture, overlay]{\\foreach \\stroke in {1,...,\\arabic{highlight}} \\draw[highlighter] (begin highlight \\stroke) -- (end highlight \\stroke);}}{}}}')
    t2.append('\n') # extra

    tikz_text = "\n".join(t2)
    
    return tikz_text
    
    
    
# find positions of tikzmarks in aux file
def find_positions(aux, marker='title', error_out=True):
    error = False
    # this part gets the marker names and aligns them to a pgfid ID
    pStarts = []; pEnds = []
    tStarts = []; tEnds = [] # pdfid counts
    astarts = []; aends=[]
    mstarts = []; mends = [] # markers
    for a in aux:
        if marker+'Start' in a and ('writefile' not in a):
            #a = a.split("}{")[1]#.split("}")[0]
            tStarts.append(a.split("pgfid")[-1].split("}")[0])
            pStarts.append("pgfid" + a.split("pgfid")[-1].split("}")[0])
            #print(a)
            astarts.append(a)
            mstarts.append(a.split(str(marker+'Start'))[-1].split('}')[0])
        elif marker+'End' in a and ('writefile' not in a):
            #a = a.split("}{")[1]#.split("}")[0]
            tEnds.append(a.split("pgfid")[-1].split("}")[0])
            pEnds.append("pgfid" + a.split("pgfid")[-1].split("}")[0])
            aends.append(a)
            mends.append(a.split(str(marker+'End'))[-1].split('}')[0])
    # if not found, try with uppercase for some reason??
    if [pStarts,pEnds,tStarts,tEnds,astarts,aends]==[[],[],[],[],[],[]]:
        for a in aux:
            if str(marker+'Start').upper() in a and ('writefile' not in a):
                #a = a.split("}{")[1]#.split("}")[0]
                tStarts.append(a.split("pgfid")[-1].split("}")[0])
                pStarts.append("pgfid" + a.split("pgfid")[-1].split("}")[0])
                #print(a)
                #print(a.split(str(marker+'Start').upper())[-1].split('}')[0])
                mstarts.append(a.split(str(marker+'Start').upper())[-1].split('}')[0])
                astarts.append(a)
            elif str(marker+'End').upper() in a and ('writefile' not in a):
                #a = a.split("}{")[1]#.split("}")[0]
                tEnds.append(a.split("pgfid")[-1].split("}")[0])
                pEnds.append("pgfid" + a.split("pgfid")[-1].split("}")[0])
                aends.append(a)
                mends.append(a.split(str(marker+'End').upper())[-1].split('}')[0])

    # checks for misalignments
    if len(pStarts) != len(pEnds):
        print('not matched starts and ends')
        #print(pStarts)
        #print(pEnds)
        print(len(pStarts), len(pEnds))
        print(len(tStarts), len(tEnds))
        for ps,pe,pps,ppe,aas,aae in zip(tStarts,tEnds,pStarts,pEnds,astarts,aends):
            ps1 = ps.split(marker+'Start')[-1]
            pe1 = pe.split(marker+'End')[-1]
            if ps1!=pe1:
                print('mismatch:', ps,pe,pps,ppe)
                print(aas)
                print(aae)
                error = True
                # return error
                return '', '', '', '', pps, ppe, '','','','',error


    # get actual x/y locations AND page location
    xstart = []; ystart = []; xend=[]; yend=[]; pageStart = []; pageEnd = []
    others = []; othersPages = []
    for a in aux:
        if 'pgfsyspdfmark' in a and 'pgfid' in a and ('writefile' not in a):
            #print(a)
            asplit = a.split('pgfid')[-1].split("}")[0]
            try: # maybe a start?
                ind = tStarts.index(asplit)
                x = a.split("pgfid"+tStarts[ind]+"}{")[-1].split("}")[0]
                y = a.split("pgfid"+tStarts[ind]+"}{")[-1].split("}{")[-1].split("}")[0]
                xstart.append(x); ystart.append(y)
                #print(a)
                #print(x,"|", y)
            except:
                #print('no')
                try:
                    ind = tEnds.index(asplit)
                    x = a.split("pgfid"+tEnds[ind]+"}{")[-1].split("}")[0]
                    y = a.split("pgfid"+tEnds[ind]+"}{")[-1].split("}{")[-1].split("}")[0]
                    xend.append(x); yend.append(y)
                except:
                    #print('issue')
                    others.append(asplit)
        elif 'savepicturepage' in a and 'pgfid' in a and ('writefile' not in a):
            asplit = a.split('pgfid')[-1].split("}")[0]
            try: # maybe a start?
                ind = tStarts.index(asplit)
                pagenum = a.split("pgfid"+tStarts[ind]+"}{")[-1].split("}")[0]
                pageStart.append(int(pagenum))
                #print(a)
                #print(x,"|", y)
            except:
                #print('no')
                try:
                    ind = tEnds.index(asplit)
                    pagenum = a.split("pgfid"+tEnds[ind]+"}{")[-1].split("}")[0]
                    pageEnd.append(int(pagenum))
                except:
                    #print('issue')
                    othersPages.append(asplit)
                    
    #checks
    if len(xstart) != len(xend) != len(ystart) != len(yend):
        if error_out:
            error_and_quit('un matched start and end')
        else:
            error = True
            return '', '', '', '', '', '', '','','','',error
    return xstart, ystart, xend, yend, pageStart, pageEnd, tStarts, tEnds, mstarts,mends,error


# collect "real" tex words from parsing tex
def find_in_tex(texfile,mstart,mend,mtype='mainBody',
                error_out=False, return_raw=False):
    """
    return_raw: return un-formatted tex too?
    """
    if type(mstart) != list: mstart = [mstart]
    if type(mend) != list: mend = [mend]
    with open(texfile,'r') as f:
        text = f.read()
    words = []
    errors = []
    raw = []
    for ms,me in zip(mstart,mend):
        err = False
        ss = '\\tikzmark{' + mtype + 'Start' + str(ms) + '}'
        se = '\\tikzmark{' + mtype + 'End' + str(me) + '}'
        if ss not in text:
            if ss.upper() in text: 
                ss = ss.upper()
            else:
                if error_out:
                    error_and_quit('no start in text!')
                else:
                    errors.append('missing starter:'+str(ss))
                    err = True
        if se not in text:
            if se.upper() in text: 
                se = se.upper()
            else:
                if error_out:
                    error_and_quit('no end in text!')
                else:
                    errors.append('missing ender:'+str(se))
                    err = True
        if not err:
            ind1 = text.index(ss) + len(ss)
            ind2 = text.index(se)
            try:
                word = LatexNodes2Text().latex_to_text(text[ind1:ind2])
            except:
                #print('roll back to raw')
                word = text[ind1:ind2]
            words.append(word)
            raw.append(text[ind1:ind2])
         
    if error_out:
        if not return_raw:
            return words
        else:
            return words, raw
    else:
        if not return_raw:
            return words, errors
        else:
            return words, raw, errors

        
        

# how many pages in a PDF?
# how many pages?
def count_pdf_pages(f):
    parsed = True
    with open(f,'rb') as ff:
        try:
            parser = PDFParser(ff)
        except:
            print('cant parse parser', f)
            parsed = False
        if parsed:
            try:
                document = PDFDocument(parser)
            except:
                print('cant parse at resolve stage', f)
                parsed = False
            if parsed:
                if resolve1(document.catalog['Pages']) is not None:
                    pages_count = resolve1(document.catalog['Pages'])['Count']  
                else:
                    pages_count = 1
    if not parsed:
        print('could not parse, assume pages = 1')
        pages = 1
    else:
        pages = pages_count
    return pages


# save all pages in a temporary directory to load and plot later
def save_pages(ffout, pages,tmp_dir, # ffout is the modified .tex file
    pdffigures_dpi = 72, # default
    fac_dpi = 4, # how much larger to make jpeg to avoid any antialiasing, check if exist 
               check_exist=True,
              save_fmt = 'jpeg', 
              fac_down_dpi = 1, 
              beginner='', 
              return_error = False): 
    
    if ffout[-4:] != '.pdf':
        pdfout = ffout.replace('.tex','.pdf')
        split_ffout = ffout.split('/')[-1].split('.tex')[0]
    else:
        pdfout = ffout
        split_ffout = ffout.split('/')[-1].split('.pdf')[0]
    
    if fac_down_dpi == None:
        fac_down_dpi = 1.0/fac_dpi

    page_errors = []
    for page in pages:
        outimgname = tmp_dir+ beginner + \
              split_ffout + \
             '_p'+str(page) + '.'+save_fmt
        #print(outimgname)
        #print(

        # check if it exists
        err = False
        if (not os.path.exists(outimgname)) or (not check_exist):
            try:
                wimgPDF = WandImage(filename=pdfout +'[' + str(int(page)) + ']', 
                                    resolution=pdffigures_dpi*fac_dpi, format='pdf') #2x DPI which shrinks later
            except:
                err = True
                print('error in DPI')
                print(pdfout)
                
            if not err:
                thisSeq = wimgPDF.sequence
                imPDF = thisSeq[0] # load PDF page into memory


                # make sure we do our best to capture accurate text/images
                imPDF.resize(width=int(round(fac_down_dpi*imPDF.width)),
                             height=int(round(fac_down_dpi*imPDF.height)))
                imPDF.background_color = Color("white")
                imPDF.alpha_channel = 'remove'
                WandImage(imPDF).save(filename=outimgname)
                del imPDF
        page_errors.append(err)
    if return_error: 
        return page_errors

    
# get page's words
def parse_page(fname_pdf, img_sizeIn, ipage, 
               xs_main,ys_main,xe_main,ye_main,
               mStart_main,mEnd_main,pStart_main, pEnd_main,
              mtype='mainBody', order_tag = 'word', return_raw = True,
              fpix=8,fac_dpi = 4, fac_down_dpi = 1):
    
    pdf_mark_order = []
    word_pdf = [] # word_pdf.copy()

    hyp = []; order_hyp = []
    
    fac = float(fac_dpi)/float(fac_down_dpi)

    for xmin,ymin,xmax,ymax,ms,me,ps,pe in zip(xs_main,ys_main,
                                               xe_main,ye_main,
                                       mStart_main,mEnd_main,
                                       pStart_main, pEnd_main):
        if ps-1 == ipage: # start and end y are different
            # to pixel page coords
            xmin=float(xmin)*fac; ymin=float(ymin)
            xmax=float(xmax)*fac; ymax=float(ymax)
            xmin = int(round(xmin/sp_printer[0])); 
            xmax = int(round(xmax/sp_printer[0]))
            #ymin = img_sizeIn[0]-int(round((ymin*fac-fpix*fac)/sp_printer[0]))
            ymin = img_sizeIn[0]-int(round(ymin*fac/sp_printer[0]))
            ymax = img_sizeIn[0]-int(round(ymax*fac/sp_printer[0]))
            xmin = round(xmin); xmax = round(xmax); 
            ymin = round(ymin); ymax = round(ymax)

            w,r,e = find_in_tex(fname_pdf,ms, me, 
                                mtype=mtype,return_raw=return_raw)
            if int(ymin)==int(ymax): # non-hyphenated
                try:
                    word_pdf.append( (xmin,ymin,xmax,ymax,w[0],r[0]) )
                    pdf_mark_order.append((order_tag,int(ms)))
                except:
                    try:
                        print('single word order:', w, r)
                        word_pdf.append( (xmin,ymin,xmax,ymax,w[0],r) )
                        pdf_mark_order.append((order_tag,int(ms)))
                    except:
                        print('single word order, missing:', w, r)
                        #word_pdf.append( (xmin,ymin,xmax,ymax,w,r) )
                    
            else: # hyphenated
                try:
                    hyp.append( (xmin,ymin,xmax,ymax,w[0],r[0]) )
                    order_hyp.append(('hyp-'+order_tag,int(ms)))
                except:
                    try:
                        print('single word order (hyphen):', w, r)
                        hyp.append( (xmin,ymin,xmax,ymax,w[0],r) )
                        order_hyp.append(('hyp-'+order_tag,int(ms)))
                    except:
                        print('single word order (hyphen), missing:', w, r)
                
    return word_pdf, pdf_mark_order, hyp, order_hyp


def get_hyp_top_bottom(hyp, regular, hyp_order, regular_order, 
                       nwords_out = 25, verbose=False, outFac = 5,
                      fpix=8):
    """
    nwords_out : gives bounding to look only around the X number of words of the hyphen
    """
    # get y's
    ys = []
    for xmin, ymin, xmax, ymax, w, r in regular:
        ys.append(ymin)
    ys = np.array(ys)

    pos = []
    for t,p in regular_order:
        pos.append(p)
    pos = np.array(pos)
    
    #aprint(len(pos), len(ys))

    hyp_top = []
    hyp_bottom = []
    
    #print('nwords out', nwords_out)

    for (xmin, ymin, xmax, ymax, w, r),(t,order) in zip(hyp,hyp_order):
        ymin_save = ymin
        ymax_save = ymax
        hypOK = True
        if verbose:
            print('')
            # check if hyp_top below hyp_bottom --> if so, it has gone to next column
            if ymin > ymax:
                print('---- wraps column ----')
            print('word :', r)
        # the data about this particular line -- where it starts
        # make sure the mark order is LESS than the hyphenated mark order
        # ymin == ys means that it is exactly on this line
        #line_before = np.array(regular,dtype=object)[np.where((ymin==ys)&(order > pos) & (order-nwords_out <= pos))]
        # ymin <= ys means it can be above
        line_before = np.array(regular,dtype=object)[np.where((ymin>=ys)&(order > pos) & (order-nwords_out <= pos))]
        if verbose:
            l = ''
            for xxmin, yymin, xxmax, yymax, ww, rr in line_before:
                l += rr + ' '
            print('   before:', l)
        # and the subsequent line
        #line_after = np.array(regular,dtype=object)[np.where((ymax==ys)&(order<pos)&(order+nwords_out >= pos))]
        line_after = np.array(regular,dtype=object)[np.where((ymax<=ys)&(order<pos)&(order+nwords_out >= pos))]
        if verbose:
            l = ''
            for xxmin, yymin, xxmax, yymax, ww, rr in line_after:
                l += rr + ' '
            print('    after:', l)
        # xmin of the whole line = xmin of line_before
        if len(line_before)>0 and len(line_after)>0:
            #xmin_before = min(line_before[:,0].min(), line_after[:,0].min())
            #xmax_after = max(line_after[:,2].max(), line_before[:,2].max())
            xmin_before = line_before[:,0].min()
            xmax_after = line_after[:,2].max()
        elif len(line_before)>0:
            xmin_before = line_before[:,0].min()
            xmax_after = line_before[:,2].max()
        elif len(line_after)>0:
            xmin_before = line_after[:,0].min()
            xmax_after = line_after[:,2].max()
        else: # neither? try again and then give up
            line_before = np.array(regular,dtype=object)[np.where((ymin>=ys)&(order > pos) & (order-nwords_out*outFac <= pos))]
            line_after = np.array(regular,dtype=object)[np.where((ymax<=ys)&(order<pos)&(order+nwords_out*outFac >= pos))]
            # xmin of the whole line = xmin of line_before
            if len(line_before)>0 and len(line_after)>0:
                xmin_before = min(line_before[:,0].min(), line_after[:,0].min())
                xmax_after = max(line_after[:,2].max(), line_before[:,2].max())
            elif len(line_before)>0:
                xmin_before = line_before[:,0].min()
                xmax_after = line_before[:,2].max()
            elif len(line_after)>0:
                xmin_before = line_after[:,0].min()
                xmax_after = line_after[:,2].max()
            else: # no luck
                hypOK = False
            
            
        #xmin_before = line_before[:,0].min()
        # xmax of the whole line = xmax of line_after(ish)
        #xmax_after = line_after[:,2].max()

        # check if hyp_top below hyp_bottom --> if so, it has gone to next column
        # skip this for now
        if ymin > ymax:
            if verbose: print('---- wrap column hyphenated not supported!!')
            hypOK = False
        # make these hyphenated PARTS of the word
        if hypOK:
            hyp_top.append( (xmin,ymin_save, xmax_after,ymin_save, w, r) )
            hyp_bottom.append( (xmin_before,ymax_save, xmax, ymax_save, w, r) )
    return hyp_top, hyp_bottom


def get_ocr_words(tree,ns, return_paragraph = False):
    words = []
    for i,item in enumerate(tree.xpath("//tei:p[@class='ocr_par']", 
                                       namespaces=ns)):
        for pw,pc,wordTree in zip(item.xpath(".//*[@class='ocrx_word']/text()",
                                             namespaces=ns), 
                               item.xpath(".//*[@class='ocrx_word']/@title",
                                          namespaces=ns),
                              item.xpath(".//*[@class='ocrx_word']", 
                                         namespaces=ns)):
            # angle too
            line = wordTree.xpath("../@title", namespaces=ns) # this should be line tag
            lineid = wordTree.xpath("../@id", namespaces=ns) # also line tag
            if return_paragraph:
                paragraph = wordTree.xpath("../../@title",namespaces=ns)
                paragraph_id = wordTree.xpath("../../@id",namespaces=ns)
            lang = wordTree.xpath("../../@lang", namespaces=ns) # this should be par tag
            if len(line) > 1:
                print('HAVE TOO MANY PARENTS')
            if 'textangle' in line[0]:
                # grab text angle
                textangle = float(line[0].split('textangle')[1].split(';')[0])
            else:
                textangle = 0.0
            # also, get fontsize
            f = float(line[0].split('x_size')[-1].split(';')[0])
            # also get baseline for the line
            if 'baseline' in line[0]:
                baseline0,baseline1 = np.array(line[0].split('baseline')[-1].split(';')[0].rstrip().lstrip().split(' ')).astype('float')
            else:
                baseline0,baseline1 = np.nan,np.nan
            # get line bounding box
            lxmin,lymin,lxmax,lymax = np.array(line[0].split('bbox')[-1].split(';')[0].rstrip().lstrip().split(' ')).astype('int')
            # line decenders and ascenders
            des = float(line[0].split('x_descenders')[-1].split(';')[0])
            asc = float(line[0].split('x_ascenders')[-1].split(';')[0])
            if len(lineid)>1: # too many!
                print('have too many lines!!')
            lineid = lineid[0]
            lineinfo = {'xmin':lxmin,'xmax':lxmax,'ymin':lymin,'ymax':lymax,
                        'fontsize':f,'baseline0':baseline0,
                        'baseline1':baseline1, 'descenders':des, 
                        'ascenders':asc, 'lineID':lineid}
            
            if return_paragraph:
                lxmin,lymin,lxmax,lymax = np.array(paragraph[0].split('bbox')[-1].rstrip().lstrip().split(' ')).astype('int')
                if len(paragraph_id)>1: # too many!
                    print('have too many paragraphs!!')
                paragraph_id = paragraph_id[0]
                paragraphinfo = {'xmin':lxmin,'xmax':lxmax,'ymin':lymin,'ymax':lymax,
                            'paragraphID':paragraph_id}
                

            bb = pc.split('bbox ')[1].split(';')[0]
            bb = np.array(bb.split(' ')).astype('int').tolist()    
            x = bb[0]; y = bb[1]
            w = bb[2]-x; h = bb[3]-y
            c = pc.split('x_wconf ')[1].split("'")[0]
            # xmin,ymin,width,height,rotation angle, confidence in %, text
            if not return_paragraph:
                where = ( ((x,y,w,h,textangle,int(c),f,lang,lineinfo),pw) )  
            else:
                where = ( ((x,y,w,h,textangle,int(c),f,lang,lineinfo, paragraphinfo),pw) )  
                
            if where not in words:
                words.append( where )   

    return words



def plot_ocr_words(imPlotIn,words, 
                   drawOCRbox = False, 
                   drawOCRwords=True,
                  color_text = (0,0,255),
                  color_box = (0,255,0),
                  font_size=18, line_width=1,
                  use_text_midpoint=False,
                  font_size_ref = 32, 
                  img_size = None,
                  fontpath='/Users/jnaiman/scienceDigitization/simsun.ttc'):
    '''
    words: assume in format of tuple of (x,y,w,h,angle,confidence,fontsize,l?),text)
    use_text_midpoint: set to true to plot words in the centers of boxes
    img_size : will rescale to this size if not None
    '''
    
    if img_size == None:
        img_size = imPlotIn.shape
    
    # now loop and save this to another image
    #imPlot = backtorgb.copy()
    #imPlot[:,:,:] = 255


    imPlot = Image.fromarray(imPlotIn)
    draw = ImageDraw.Draw(imPlot)

    if drawOCRwords:
        for (x,y,w,h,rot,conf,f,l),text in words:
            fs = int(round(f/font_size_ref*font_size))
            font = ImageFont.truetype(fontpath, fs)
            if use_text_midpoint:
                ###x = x + 0.5*w
                y = y + 0.5*h

            # shape = (y,x)
            x = int(round(x*imPlotIn.shape[1]/img_size[1]))
            y = int(round(y*imPlotIn.shape[0]/img_size[0]))
            w = int(round(w*imPlotIn.shape[1]/img_size[1]))
            h = int(round(h*imPlotIn.shape[0]/img_size[0]))
            if len(text) > 0:
                if rot == 0:
                    draw.text( (x, y), text, font=font, fill=color_text)
                else:
                    img_txt = Image.new('RGBA', font.getsize(text), color=(255,255,255,1))
                    draw_txt = ImageDraw.Draw(img_txt)
                    draw_txt.text( (0,0), text, font=font, fill=color_text)
                    rot_im = img_txt.rotate(rot, expand=True, fillcolor="white")
                    imPlot.paste(rot_im, (x,y))

    imPlot = np.array(imPlot)    

    if drawOCRbox:
        for (x,y,w,h,rot,conf,f,l),text in words:
            # shape = (y,x)
            x = int(round(x*imPlotIn.shape[1]/img_size[1]))
            y = int(round(y*imPlotIn.shape[0]/img_size[0]))
            w = int(round(w*imPlotIn.shape[1]/img_size[1]))
            h = int(round(h*imPlotIn.shape[0]/img_size[0]))
            cv.rectangle( imPlot, (x, y), (x+w, y+h), color_box, line_width )
            
    return imPlot


def iou_calc(x1, y1, w1, h1, x2, y2, w2, h2, return_individual = False): 
    '''
    Calculate IOU between box1 and box2

    Parameters
    ----------
    - x, y : box ***center*** coords
    - w : box width
    - h : box height
    - return_individual: return intersection, union and IOU? default is False
    
    Returns
    -------
    - IOU
    '''   
    xmin1 = x1 - 0.5*w1
    xmax1 = x1 + 0.5*w1
    ymin1 = y1 - 0.5*h1
    ymax1 = y1 + 0.5*h1
    xmin2 = x2 - 0.5*w2
    xmax2 = x2 + 0.5*w2
    ymin2 = y2 - 0.5*h2
    ymax2 = y2 + 0.5*h2
    interx = np.minimum(xmax1, xmax2) - np.maximum(xmin1, xmin2)
    intery = np.minimum(ymax1, ymax2) - np.maximum(ymin1, ymin2)
    inter = interx * intery
    union = w1*h1 + w2*h2 - inter
    iou = inter / (union + 1e-6)
    if not return_individual:
        return iou
    else:
        return inter, union, iou

    
    
#    words = words_ocr.copy()
#    verbose = True
def align_sgt_ocr(words, word_pdf_full, pdf_mark_order_full, verbose=True, fpix=8, fsize=11):

    #alignment_pages = {} #: for all pdf words dictionary, one for each page

    # ------- OCR words in to format of PDF -------
    # get all OCR words into PDF axis for faster comparison
    word_inds = np.arange(len(words))
    ocr_boxes = []
    for oi in range(len(words)):
        word = words[oi][0] # word info
        #text_ocr_here = words[oi][-1] # default is *not* raw -- with greek/latin
        xmin2,ymin2,w,h = word[0],word[1],word[2],word[3]
        xmax2 = xmin2+w; ymax2 = ymin2+h
        ocr_boxes.append((xmin2,ymin2,xmax2,ymax2))
    ocr_boxes = np.array(ocr_boxes)


    # *********** Align OCR to PDF words *************
    ocr_words_aligned = [] # length of PDF words
    ocr_inds = np.arange(len(words))

    # keep track of OCR-matched indices
    # in theory there should only be ONE pdf word for each OCR word
    ocr_inds_track = []

    pdf_words_aligned = []; pdf_words_aligned_type = []
    if verbose==2: 
        print('matching', len(ocr_boxes), 'ocr boxes,', len(word_pdf_full), 'pdf boxes')
    cond1=''; cond2=''; cond3=''; cond4=''
    if len(ocr_boxes) > 0:
        for ind in range(len(word_pdf_full)): # all PDF words
            pdf_words_aligned.append(word_pdf_full[ind])
            pdf_words_aligned_type.append(pdf_mark_order_full[ind][0])
            # find all overlapping -- all OCR words that overlap with the PDF word
            xmin1,ymin1,xmax1,ymax1 = word_pdf_full[ind][:4]
            cond1 = xmin1>ocr_boxes[:,2]
            cond2 = xmax1<ocr_boxes[:,0]
            cond3 = ymax1<ocr_boxes[:,1]
            cond4 = ymin1-fpix>ocr_boxes[:,3]
            cond = cond1 | cond2 | cond3 | cond4
            # opposite
            oinds = ocr_inds[~cond]
            words_this_pdf = []; xs = []
            text_ocr = []; oinds_save = []
            for ioi,oi in enumerate(oinds): # loop through all of the OCR words that overlap with this PDF word
                xmin2,ymin2,xmax2,ymax2 = ocr_boxes[oi]

                # is it already in there? i.e. have we already matched this OCR word to a PDF word before?
                useOCRWord = True
                if oi in ocr_inds_track: # it is already in there! 
                    # We have to figure out which match to keep -- the initially tracked word, or this new one?
                    # we have to search through all already stored words... there is probably a better way
                    oasave = -1
                    for ioa,oa in enumerate(ocr_words_aligned): # find which word matches
                        if oi in oa[-1]: # in stored list of inds?
                            oasave = oa # saved info about stuff -- there will always only be 1!
                            break
                    if oasave == -1: # this shouldn't happen??
                        print('   oi not found=', oi) 
                        import sys; sys.exit()
                    else: # have a thing!
                        if verbose==2: print('overlap for', oi)
                        indoverlap = ocr_words_aligned.index(oasave) # what PDF word does this OCR word overlap with that we have already stored
                        xs_here = []; oinds_here = []
                        # for this new OCR word -- does it overlap more with the prior PDF word (indoverlap), or the new one (ind)
                        # get the bounding box info for the prior PDF box & this box
                        #           [NEW pdf word,            OLD pdf word]
                        pdfboxes = [word_pdf_full[ind][:4],word_pdf_full[indoverlap][:4]]
                        # get OCR box for this current box
                        ocr_boxes_here = ocr_boxes[oi] # new bounding box this OCR word, grabbed again
                        # calculate IOUs of OCR word and both PDF boxes
                        ious = []
                        for pb in pdfboxes:
                            ious.append(iou_calc(pb[0], pb[1], pb[2], pb[3],
                                                ocr_boxes_here[0],ocr_boxes_here[1],
                                                ocr_boxes_here[1],ocr_boxes_here[3]))
                        # take the original or the found
                        if np.argmax(ious) == 1: # the previously found PDF word matches best with this OCR
                            if verbose==2: print('taking the previous word, will ignore this OCR word for this PDF')
                            useOCRWord = False # nothing changes
                        else: # the NEW PDF word is the better match
                            if verbose==2: print('take new PDF word!')
                            # ocr_words_aligned == what words are aligned with the PDF word (ind)
                            if len(ocr_words_aligned[indoverlap][0]) > 1: # we have multiple OCR words attached to this PDF! must take ONLY the matched one here
                                # find the one that matches in the collection, and remove ONLY that OCR word
                                if verbose==2: print('   multi words!')
                                ocr_words_aligned_out = []
                                texts2 = []; wtp2 = []; xs2 = []; oinds2 = []
                                for iw,(w,winfo,w2,w3) in enumerate(zip(ocr_words_aligned[indoverlap][0], ocr_words_aligned[indoverlap][1],
                                                                       ocr_words_aligned[indoverlap][2],ocr_words_aligned[indoverlap][3])):
                                    # check that the words match
                                    # is the found word the same as the initial word? if so, we want to make
                                    #  sure it is removed from the prior PDF word it was attached to
                                    if w == words[oi][1] and ( (winfo == list(words[oi][0])) or (winfo == tuple(words[oi][0])) ): # formatting sillyness
                                        if verbose==2: print('found', w, 'ind=',ind,'oi=', oi, 'words[oi]=', words[oi][1])
                                        pass
                                    else: # don't match, then keep
                                        texts2.append(w); wtp2.append(winfo)
                                        xs2.append(w2); oinds2.append(w3)
                                ocr_words_aligned[indoverlap] = (texts2, wtp2, xs2, oinds2)

                            else: # only 1 OCR word associated with prior PDF word, set to empty
                                if verbose==2: print('   only one!')
                                if verbose==2: print(ocr_words_aligned[indoverlap])
                                ocr_words_aligned[indoverlap] = ('', words_this_pdf,xs, [-2])

                if useOCRWord:
                    word = words[oi][0] # word info
                    text_ocr_here = words[oi][-1] # default is *not* raw -- with greek/latin
                    words_this_pdf.append(word)
                    text_ocr.append(text_ocr_here)
                    xs.append((xmin2,ymin2,xmax2,ymax2))
                    ocr_inds_track.append(oi)
                    ocr_inds_track = np.unique(ocr_inds_track).tolist() # don't double count, shouldn't be an issue
                    oinds_save.append(oi)
                else:
                    pass

            # 3. if multiple OCR words for a PDF word, order by x and smoosh into 1 word
            if len(xs)>1: # many words
                asort = np.argsort(np.array(xs)[:,0]) # sort by xmin
                words_this_pdf = np.array(words_this_pdf,dtype=object)[asort]
                texts = np.array(text_ocr)[asort].astype('str').tolist()
                ocr_words_aligned.append((texts, words_this_pdf.tolist(),xs, np.array(oinds_save)[asort].tolist()))
            elif len(xs) == 0: # no words!
                if verbose==2: print('no OCR words match this PDF! : ', word_pdf_full[ind])
                ocr_words_aligned.append(('', words_this_pdf,xs, [-2]))
            else: # one word
                ocr_words_aligned.append((text_ocr, words_this_pdf,xs, [oinds_save[0]]))


        # 4. "unwrap" PDF words by tikzmark number and unwrap OCR word's associated
        # align
        if len(pdf_mark_order_full)>0:
            pdf_mark_order_full = np.array(pdf_mark_order_full, dtype=object)
            asort = np.argsort(pdf_mark_order_full[:,1])
            if len(pdf_words_aligned) > 0 and len(pdf_words_aligned_type)>0 and len(ocr_words_aligned)>0:
                pdf_words_aligned = np.array(pdf_words_aligned,dtype=object)[asort].tolist()
                pdf_words_aligned_type = np.array(pdf_words_aligned_type,dtype=object)[asort].tolist()
                ocr_words_aligned = np.array(ocr_words_aligned,dtype=object)[asort].tolist()

    # ********** align -- make dictionaries ***************
    alignment = {}
    # NON-hypenated!
    for pmark_all,pword,ptype,oword in zip(pdf_mark_order_full, 
                                       pdf_words_aligned,pdf_words_aligned_type,
                                       ocr_words_aligned):
        pmark = pmark_all[-1] # location index
        # PDF
        pwd = {}
        if ptype != 'word':
            pwd['word'] = pword[-1]
        else:
            pwd['word'] = pword[-2]
        pwd['raw'] = pword[-1]
        pwd['xmin'] = pword[0]; pwd['ymin'] = pword[1]-fpix
        pwd['xmax'] = pword[2]; pwd['ymax'] = pword[3]
        if 'hyp-' not in ptype:
            pwd['hyphenated'] = False
        else:
            pwd['hyphenated'] = True
        pwd['assumed fontsize transformations'] = {'fontsize':fsize, 'fpix':fpix}
        pwd['word type'] = ptype

        # ocr -- could be multiple words per PDF! already sorted by x
        owd = {}
        for o,wtp,x in zip(oword[0],oword[1],oword[2]): # all words
            ow = {}
            # locations, scaled to PDF 
            ow['xmin'] = x[0]; ow['ymin']=x[1]; ow['xmax']=x[2]; ow['ymax']=x[3]
            ow['rotation'] = wtp[4]
            ow['confidence'] = wtp[5]
            ow['font size'] = wtp[6]
            ow['languages'] = wtp[7]
            ow['baseline0'] = wtp[8]['baseline0']
            ow['baseline1'] = wtp[8]['baseline1']
            ow['descenders'] = wtp[8]['descenders']
            ow['ascenders'] = wtp[8]['ascenders']
            ow['lineID'] = wtp[8]['lineID']
            owd[o] = ow # o is the 

        alignment[pmark] = {'PDF word':pwd, 'OCR words':owd}
        #alignment_pages['page'+str(ipage)] = alignment

        # ******* Getting ignore/not ignore *******
        pdf_text = ''; ocr_text = ''; pdf_mark_char = []
        pdfmarks = np.sort(list(alignment.keys()))
        ignore = []
        for pm in pdfmarks: # loop over all words
            pdf_text += alignment[pm]['PDF word']['word'] + ' '
            pdf_mark_char_here = []; ignore_here = []
            for p in range(len(alignment[pm]['PDF word']['word'])+1): 
                pdf_mark_char_here.append(pm)
            pdf_mark_char.append(pdf_mark_char_here)
            # all OCR words
            for ow in alignment[pm]['OCR words'].keys():
                ocr_text += ow + ' '
            if alignment[pm]['PDF word']['word type'] == 'word':
                for p in range(len(alignment[pm]['PDF word']['word'])+1): 
                    ignore_here.append(' ')
            elif alignment[pm]['PDF word']['word type'] == 'inline':
                for p in range(len(alignment[pm]['PDF word']['word'])+1): 
                    ignore_here.append('I')
                if verbose==2: print('ignoring inline:', alignment[pm]['PDF word']['word'])
            elif alignment[pm]['PDF word']['word type'] == 'cite':
                for p in range(len(alignment[pm]['PDF word']['word'])+1): 
                    ignore_here.append('C')
                if verbose==2: print('ignoring citation:', alignment[pm]['PDF word']['word'])
            elif alignment[pm]['PDF word']['word type'] == 'ref':
                for p in range(len(alignment[pm]['PDF word']['word'])+1): 
                    ignore_here.append('R')
                if verbose==2: print('ignoring ref:', alignment[pm]['PDF word']['word'])
            elif 'hyp-' in alignment[pm]['PDF word']['word type']:
                for p in range(len(alignment[pm]['PDF word']['word'])+1): 
                    ignore_here.append('H')
                if verbose==2: print('ignoring hyphen:', alignment[pm]['PDF word']['word'])  
            else: # this shold not happen
                for p in range(len(alignment[pm]['PDF word']['word'])+1): 
                    ignore_here.append('*')
                if verbose==2: print('ignoring ???????:', alignment[pm]['PDF word']['word'])
            ignore.append(ignore_here)
            
    return alignment
