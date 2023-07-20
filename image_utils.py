from wand.image import Image as WandImage
from wand.color import Color


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
