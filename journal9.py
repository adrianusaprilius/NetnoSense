"""NetnoSense v9: deterministic journal preparation, no generative API.
All new data lives in the project's existing JSONB payload. No network calls.
"""
import copy, hashlib, io, json, re, zipfile
from datetime import datetime, timezone
from urllib.parse import urlsplit
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

CATALOG = [
 ('International Journal of Contemporary Hospitality Management','Emerald','hospitality restaurant tourism service marketing','https://www.emeraldgrouppublishing.com/journal/ijchm'),
 ('Journal of Retailing and Consumer Services','Elsevier','retail consumer behavior digital marketing service','https://www.sciencedirect.com/journal/journal-of-retailing-and-consumer-services'),
 ('Journal of Business Research','Elsevier','marketing management business consumer behavior','https://www.sciencedirect.com/journal/journal-of-business-research'),
 ('International Journal of Hospitality Management','Elsevier','hospitality restaurant service tourism','https://www.sciencedirect.com/journal/international-journal-of-hospitality-management'),
 ('Asia Pacific Journal of Marketing and Logistics','Emerald','marketing logistics consumer behavior asia','https://www.emeraldgrouppublishing.com/journal/apjml'),
 ('Journal of Consumer Marketing','Emerald','consumer behavior marketing','https://www.emeraldgrouppublishing.com/journal/jcm'),
 ('Journal of Consumer Behaviour','Wiley','consumer behavior psychology marketing','https://onlinelibrary.wiley.com/journal/14791838'),
 ('Psychology & Marketing','Wiley','psychology consumer behavior marketing','https://onlinelibrary.wiley.com/journal/15206793'),
 ('Journal of Ethnic Foods','Springer Nature','food culinary culture netnography','https://journalofethnicfoods.biomedcentral.com/'),
]
FIELDS = ['title','authors','abstract','keywords','introduction','literature','question','methods','findings','discussion','contribution','implications','limitations','conclusion','ethics','availability','funding','ai_disclosure']
BODY = FIELDS[4:]
EXTRA = ['abstract_purpose','abstract_design','abstract_findings','abstract_originality','cover','highlights','credit','competing','corresponding','gap_evidence','novelty_evidence']
HEADINGS = {'introduction':'Introduction','literature':'Literature review','question':'Research questions','methods':'Methods','findings':'Results','discussion':'Discussion','contribution':'Theoretical contributions','implications':'Managerial implications','limitations':'Limitations','conclusion':'Conclusion','ethics':'Ethical considerations','availability':'Data availability','funding':'Funding and conflicts of interest','ai_disclosure':'AI and software disclosure'}
HEADINGS_ID = dict(zip(BODY,['Pendahuluan','Kajian literatur','Pertanyaan penelitian','Metode','Hasil','Pembahasan','Kontribusi teoretis','Implikasi manajerial','Keterbatasan','Kesimpulan','Etika penelitian','Ketersediaan data','Pendanaan dan konflik kepentingan','Pengungkapan AI dan perangkat lunak']))
STRUCT = [('abstract_purpose','Purpose'),('abstract_design','Design/methodology/approach'),('abstract_findings','Findings'),('abstract_originality','Originality/value')]
TOKEN = re.compile(r'\[\[([A-Za-z0-9_.-]{1,64})\]\]')

def default_profile():
    return dict(journal='',publisher='',url='',verified=False,verified_date='',language='English',citation='APA-like',abstract_type='Unstructured',abstract_max=0,word_max=0,keywords_max=0,blind=True,highlights=False,cover=False,font='Times New Roman',size=12,heading_size=13,title_size=16,margin_top=1.0,margin_bottom=1.0,margin_left=1.0,margin_right=1.0,page_width=8.27,page_height=11.69,spacing=1.5,columns=1,template_name='',template_hash='',guideline_text='',guideline_name='',required=list(BODY),order=list(BODY),heading_overrides={})

def defaults():
    return {'profile':default_profile(),'extras':{},'citation_labels':{},'history':[]}

def clean_text(x,limit=50000):
    if not isinstance(x,str) or len(x)>limit: raise ValueError('Teks metadata jurnal tidak valid atau terlalu panjang.')
    # XML 1.0 and Word compatible characters
    return re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\ud800-\udfff\ufffe\uffff]','',x)

def profile_decode(v):
    p=default_profile()
    if not isinstance(v,dict): raise ValueError('Profil jurnal tidak valid.')
    for k in p:
        if k not in v: continue
        x=v[k]; base=p[k]
        if isinstance(base,bool):
            if not isinstance(x,bool): raise ValueError(k)
        elif isinstance(base,(int,float)):
            if isinstance(x,bool) or not isinstance(x,(int,float)) or not 0<=x<=1000000: raise ValueError(k)
        elif isinstance(base,str): x=clean_text(x,80000 if k=='guideline_text' else 1000)
        elif isinstance(base,list):
            if not isinstance(x,list) or any(y not in BODY for y in x) or len(x)!=len(set(x)): raise ValueError(k)
        elif isinstance(base,dict):
            if not isinstance(x,dict) or any(y not in BODY for y in x): raise ValueError(k)
            x={y:clean_text(z,150) for y,z in x.items()}
        p[k]=x
    for k,allowed in [('language',['English','Indonesia']),('citation',['APA-like','Harvard-like','Numeric']),('abstract_type',['Unstructured','Structured'])]:
        if p[k] not in allowed: raise ValueError(k)
    for k in ['margin_top','margin_bottom','margin_left','margin_right']:
        if not .2<=p[k]<=3: raise ValueError(k)
    for k in ['abstract_max','word_max','keywords_max','columns']:
        if int(p[k])!=p[k]: raise ValueError(k)
        p[k]=int(p[k])
    for k in ['size','heading_size','title_size']:
        if not 6<=p[k]<=32: raise ValueError(k)
    if not 1<=p['spacing']<=3 or not 5<=p['page_width']<=15 or not 7<=p['page_height']<=20 or p['columns'] not in (1,2): raise ValueError('Ukuran halaman/spasi/kolom tidak valid.')
    if p['page_width']-p['margin_left']-p['margin_right']<3 or p['page_height']-p['margin_top']-p['margin_bottom']<3: raise ValueError('Area halaman terlalu kecil.')
    if not set(p['required'])<=set(p['order']): raise ValueError('Bagian wajib harus berada dalam urutan ekspor.')
    return p

def text_map(x,fields=None,limit=50000):
    if not isinstance(x,dict) or len(x)>2000 or any(not isinstance(k,str) or len(k)>64 or (fields is not None and k not in fields) for k in x): raise ValueError('Metadata naskah tidak valid.')
    return {k:clean_text(v,limit) for k,v in x.items()}

def decode(v):
    if not isinstance(v,dict): raise ValueError('Metadata v9 tidak valid.')
    result=defaults(); result['profile']=profile_decode(v.get('profile',{})); result['extras']=text_map(v.get('extras',{}),EXTRA); result['citation_labels']=text_map(v.get('citation_labels',{}),limit=200)
    h=v.get('history',[])
    if not isinstance(h,list) or len(h)>20: raise ValueError('Riwayat dibatasi 20 revisi.')
    for r in h:
        if not isinstance(r,dict): raise ValueError('Riwayat tidak valid.')
        result['history'].append({'id':clean_text(r.get('id',''),64),'utc':clean_text(r.get('utc',''),80),'note':clean_text(r.get('note',''),500),'basis':clean_text(r.get('basis',''),128),'manuscript':text_map(r.get('manuscript',{}),FIELDS),'extras':text_map(r.get('extras',{}),EXTRA),'profile':profile_decode(r.get('profile',{})),'citation_labels':text_map(r.get('citation_labels',{}),limit=200)})
    return result

def basis(data,publication):
    return hashlib.sha256(json.dumps([data,{k:v for k,v in publication.items() if k not in ('manuscript','checks','checks_basis')}],sort_keys=True,ensure_ascii=False,default=str).encode()).hexdigest()

def add_history(v,m,note,stamp):
    q=copy.deepcopy(v); h=q['history']; now=datetime.now(timezone.utc).isoformat(); rid=hashlib.sha256((now+note).encode()).hexdigest()[:16]
    h.append(dict(id=rid,utc=now,note=note,basis=stamp,manuscript=copy.deepcopy(m),extras=copy.deepcopy(q['extras']),profile=copy.deepcopy(q['profile']),citation_labels=copy.deepcopy(q['citation_labels'])))
    q['history']=h[-20:]; return decode(q)

def find_journals(query):
    synonyms={'pemasaran':'marketing','konsumen':'consumer','perilaku':'behavior','kuliner':'food','restoran':'restaurant','pariwisata':'tourism','manajemen':'management','digital':'digital','netnografi':'netnography'}
    terms=[synonyms.get(t,t) for t in re.findall(r'\w+',query.lower()) if len(t)>2]
    rows=[]
    for name,pub,tags,url in CATALOG:
        text=(name+' '+tags).lower(); score=sum(t in text for t in set(terms))
        rows.append(dict(Jurnal=name,Penerbit=pub,Topik=tags,Kecocokan_kata=score,Situs=url))
    return sorted(rows,key=lambda r:(-r['Kecocokan_kata'],r['Jurnal']))

def safe_docx(blob):
    if len(blob)>5_000_000: raise ValueError('DOCX maksimal 5 MB.')
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        if len(z.infolist())>2000 or sum(x.file_size for x in z.infolist())>40_000_000: raise ValueError('DOCX terlalu besar setelah ekstraksi.')
        if 'word/document.xml' not in z.namelist() or any(x.lower().endswith('vbaproject.bin') for x in z.namelist()): raise ValueError('Gunakan DOCX biasa, bukan file macro.')
    return Document(io.BytesIO(blob))

def template_profile(blob,name,p):
    d=safe_docx(blob); q=copy.deepcopy(p); s=d.sections[0]
    for key,attr in [('margin_top','top_margin'),('margin_bottom','bottom_margin'),('margin_left','left_margin'),('margin_right','right_margin'),('page_width','page_width'),('page_height','page_height')]:
        value=getattr(s,attr,None)
        if value is not None: q[key]=round(value.inches,3)
    normal=d.styles['Normal']; font=normal.font
    if font.name: q['font']=font.name
    if font.size: q['size']=font.size.pt
    for k,style in [('heading_size','Heading 1'),('title_size','Title')]:
        if style in d.styles and d.styles[style].font.size: q[k]=d.styles[style].font.size.pt
    sp=normal.paragraph_format.line_spacing
    if isinstance(sp,float) and 1<=sp<=3: q['spacing']=sp
    cols=s._sectPr.find(qn('w:cols')); q['columns']=min(2,max(1,int(cols.get(qn('w:num'),'1')))) if cols is not None else 1
    q.update(template_name=name[:200],template_hash=hashlib.sha256(blob).hexdigest(),verified=False,verified_date='')
    return profile_decode(q),[x.text for x in d.paragraphs if x.text.strip()][:30]

def import_manuscript(blob):
    """Extract paragraphs by recognized heading only; no guessing from raw prose."""
    doc=safe_docx(blob); aliases={}
    for k in BODY:
        for name in (HEADINGS[k],HEADINGS_ID[k]): aliases[name.casefold()]=k
    aliases.update({'abstract':'abstract','abstrak':'abstract','keywords':'keywords','kata kunci':'keywords','introduction and research gap':'introduction','pendahuluan dan kesenjangan penelitian':'introduction','theoretical background':'literature','landasan teoretis':'literature','landasan teori':'literature','research methodology':'methods','metode penelitian':'methods','methodology':'methods','results and findings':'findings','hasil penelitian':'findings','hasil dan pembahasan':'findings','discussion and implications':'discussion','kesimpulan dan saran':'conclusion','references':'skip','daftar pustaka':'skip','bibliography':'skip','references and sources':'skip','referensi':'skip','title':'title','judul':'title','authors and affiliations':'authors','penulis dan afiliasi':'authors'})
    result={}; unassigned=[]; active=None
    for par in doc.paragraphs:
        t=par.text.strip()
        if not t: continue
        style=par.style.name if par.style else ''
        key=re.sub(r'^\s*\d+(?:\.\d+)*[.)]?\s+','',t).strip(' :').casefold()
        if style=='Title' and 'title' not in result: result['title']=t; active=None; continue
        if style.startswith('Heading') or key in aliases:
            if key in aliases: active=aliases[key]; continue
            # Preserve subheading in its current section, but do not classify new level-1 headings.
            if style in ('Heading 1','Heading 2') and style=='Heading 1': active=None
        kw=re.match(r'^(?:keywords|kata kunci)\s*:\s*(.+)$',t,re.I)
        if kw: result['keywords']=kw[1]; active=None; continue
        if active=='skip': continue
        if active: result[active]=(result.get(active,'')+'\n'+t).strip()[:50000]
        else: unassigned.append(t[:500])
    return result,unassigned[:50]

def extract_guidelines(blob,name):
    if len(blob)>5_000_000: raise ValueError('Pedoman maksimal 5 MB.')
    if name.lower().endswith('.docx'):
        d=safe_docx(blob); text='\n'.join([x.text for x in d.paragraphs]+[' | '.join(c.text for c in r.cells) for t in d.tables for r in t.rows])
    elif name.lower().endswith('.pdf'):
        from pypdf import PdfReader
        r=PdfReader(io.BytesIO(blob));
        if r.is_encrypted: raise ValueError('PDF terenkripsi tidak didukung.')
        text='\n'.join((p.extract_text() or '') for p in r.pages[:50])
    else: text=blob.decode('utf-8-sig')
    if not text.strip(): raise ValueError('Tidak ada teks terbaca; salin pedoman secara manual (tanpa OCR).')
    return clean_text(text[:80000],80000)

def guideline_hints(text):
    # Lines are evidence, not inferred requirements. Human must confirm.
    terms=r'abstract|word limit|words|keywords|font|margin|spacing|references|citation|blind|highlight|cover letter|abstrak|kata kunci|spasi|sitasi|huruf|anonim|pendanaan'
    return [s.strip()[:500] for s in re.split(r'\n|(?<=[.!?])\s+',text) if re.search(terms,s,re.I)][:60]

def count_words(s): return len(re.findall(r"\b[\w]+(?:[-’'][\w]+)*\b",s))
def abstract_text(m,v):
    if v['profile']['abstract_type']=='Structured':
        labels=STRUCT if v['profile']['language']=='English' else list(zip([k for k,_ in STRUCT],['Tujuan','Desain/metode/pendekatan','Temuan','Orisinalitas/nilai']))
        return '\n'.join(f'{label}: {v["extras"].get(key,"")}' for key,label in labels)
    return m.get('abstract','')
def ref_ok(r): return bool(r.get('Terverifikasi') and all(str(r.get(k,'')).strip() for k in ['Penulis_APA','Tahun','Judul','Sumber_APA']))
def ref_line(r,style):
    a=r['Penulis_APA'].strip(); y=r['Tahun'].strip(); t=r['Judul'].rstrip('. '); s=r['Sumber_APA'].strip(); u=r.get('DOI_URL','').strip()
    if u.startswith('10.'): u='https://doi.org/'+u
    if style=='Harvard-like': return f'{a} ({y}), “{t}”, {s}. {u}'.strip()
    if style=='Numeric': return f'{a}. {t}. {s}. {y}. {u}'.strip()
    return f'{a} ({y}). {t}. {s}. {u}'.strip()

def citation_plan(m,v,refs,include_findings=True):
    profile=v['profile']; style=profile['citation']; ref={r['ID_Referensi']:r for r in refs if ref_ok(r)}
    texts=[m.get('title',''),abstract_text(m,v),m.get('keywords','')]+[m.get(k,'') for k in profile['order'] if include_findings or k!='findings']
    ids=list(dict.fromkeys(t for s in texts for t in TOKEN.findall(s))); unknown=[i for i in ids if i not in ref]; ordered=[i for i in ids if i in ref]
    if style!='Numeric': ordered.sort(key=lambda i:(ref[i]['Penulis_APA'].casefold(),ref[i]['Tahun'],ref[i]['Judul']))
    labels={}; missing=[]
    for i in ordered:
        if style=='Numeric': labels[i]=f'[{ordered.index(i)+1}]'
        else:
            # Explicit surname/year citation is safer than guessing author list parsing.
            label=v['citation_labels'].get(i,'').strip()
            if not label: missing.append(i); labels[i]='[[ '+i+' — label sitasi belum diisi ]]'
            else: labels[i]='('+label.strip('()')+')'
    if style!='Numeric':
        duplicate={x for x in labels.values() if list(labels.values()).count(x)>1}
        missing.extend(i for i,x in labels.items() if x in duplicate and i not in missing)
    def convert(s): return TOKEN.sub(lambda x:labels.get(x[1],'[[ '+x[1]+' — referensi belum terverifikasi ]]'),clean_text(s))
    bibliography=[(f'[{j+1}] ' if style=='Numeric' else '')+ref_line(ref[i],style) for j,i in enumerate(ordered)]
    return convert,bibliography,unknown,missing,[i for i in ref if i not in ids]

def checks(m,v,refs,n,checks_valid,mapping,theme_n):
    p=v['profile']; e=v['extras']; rows=[]
    def add(name,ok,detail): rows.append(dict(Pemeriksaan=name,Status='OK' if ok else 'PERIKSA',Detail=detail))
    add('Profil jurnal',bool(p['journal'] and p['url'] and p['verified'] and p['verified_date']),'Konfirmasi pedoman terbaru oleh peneliti, bukan verifikasi otomatis.')
    add('Judul dan identitas',bool(m.get('title','').strip() and m.get('authors','').strip()),'Identitas sesuai afiliasi sebenarnya.')
    abst=abstract_text(m,v); ok=bool(m.get('abstract','').strip()) if p['abstract_type']=='Unstructured' else all(e.get(k,'').strip() for k,_ in STRUCT)
    add('Abstrak',ok and (not p['abstract_max'] or count_words(abst)<=p['abstract_max']),f'{count_words(abst)} kata (batas {p["abstract_max"] or "belum diatur"}).')
    kw=[x for x in re.split(r'[,;\n]+',m.get('keywords','')) if x.strip()]; add('Kata kunci',bool(kw) and (not p['keywords_max'] or len(kw)<=p['keywords_max']),f'{len(kw)} kata kunci.')
    missing=[k for k in p['required'] if not m.get(k,'').strip()]; add('Bagian wajib',not missing,'Belum diisi: '+', '.join(missing) if missing else 'Bagian terisi; substansi belum dinilai.')
    words=count_words(' '.join([m.get('title',''),abst,m.get('keywords','')]+[m.get(k,'') for k in p['order']])); add('Batas kata (teks peneliti)',not p['word_max'] or words<=p['word_max'],f'{words} kata; tabel, narasi otomatis, referensi dan catatan belum dihitung. Periksa di Word.')
    _,bib,unknown,labels,uncited=citation_plan(m,v,refs)
    add('Sitasi ID',not unknown and not labels and bool(bib),'ID belum sah: '+', '.join(unknown)+'; label kosong/duplikat: '+', '.join(labels)+'; referensi terpakai: '+str(len(bib))+'. Sitasi tanpa token diperiksa manual.')
    add('Data tervalidasi',n>0,f'{n} komentar memenuhi cakupan publikasi v8; tidak mengukur kecukupan sampel.')
    add('Tema manusia',theme_n>0,f'{theme_n} tema terisi; bukan bukti kualitas interpretasi.')
    add('Pemetaan teori',not mapping and theme_n>0,'Periksa kesesuaian tema, ID bukti dan referensi di menu 11.')
    add('Research gap dan novelty',bool(e.get('gap_evidence','').strip() and e.get('novelty_evidence','').strip()),'Catatan perbandingan literatur yang ditulis peneliti; tidak menilai orisinalitas.')
    add('Checklist v8',checks_valid,'Semua checklist v8 harus dikonfirmasi atas data dan naskah terkini.')
    add('Cover letter',not p['cover'] or bool(e.get('cover','').strip() and '[Explain' not in e.get('cover','') and '[Confirm' not in e.get('cover','')),'Isi dan konfirmasi bila diwajibkan.')
    add('Highlights',not p['highlights'] or bool(e.get('highlights','').strip()),'Gunakan temuan tervalidasi dan periksa batas karakter jurnal.')
    return rows

def make_doc(p):
    d=Document(); s=d.sections[0]; s.page_width=Inches(p['page_width']); s.page_height=Inches(p['page_height'])
    for k in ('top','bottom','left','right'): setattr(s,k+'_margin',Inches(p['margin_'+k]))
    cols=s._sectPr.find(qn('w:cols'))
    if cols is None: cols=OxmlElement('w:cols'); s._sectPr.append(cols)
    cols.set(qn('w:num'),str(p['columns'])); cols.set(qn('w:space'),'360')
    for name,size in [('Normal',p['size']),('Title',p['title_size']),('Heading 1',p['heading_size']),('Heading 2',p['heading_size'])]:
        style=d.styles[name]; style.font.name=p['font']; style.font.size=Pt(size); style.font.color.rgb=None; style.paragraph_format.line_spacing=p['spacing']; style.paragraph_format.space_after=Pt(6)
    d.core_properties.author=''; d.core_properties.last_modified_by=''; d.core_properties.subject=''; d.core_properties.comments=''; d.core_properties.title=''
    return d

def doc_bytes(d):
    b=io.BytesIO(); d.save(b); return b.getvalue()

def add_text(d,s):
    for par in clean_text(s).split('\n'):
        if par.strip(): d.add_paragraph(par)

def add_table(d,title,frame):
    if frame.empty: return
    d.add_heading(title,2); t=d.add_table(rows=1,cols=len(frame.columns)); t.style='Table Grid'; t.alignment=WD_TABLE_ALIGNMENT.CENTER
    for c,k in zip(t.rows[0].cells,frame.columns): c.text=clean_text(str(k)); c.paragraphs[0].runs[0].bold=True
    t.rows[0]._tr.get_or_add_trPr().append(OxmlElement('w:tblHeader'))
    for row in frame.itertuples(index=False,name=None):
        for c,x in zip(t.add_row().cells,row): c.text=clean_text(f'{x:.4f}' if isinstance(x,float) else str(x))
    for row in t.rows:
        for c in row.cells:
            for para in c.paragraphs:
                para.paragraph_format.line_spacing=1; para.paragraph_format.space_after=Pt(2)
                for run in para.runs: run.font.size=Pt(8)

def render(m,v,refs,analysis,blind=False,ready=False):
    p=v['profile']; en=p['language']=='English'; e=v['extras']; d=make_doc(p); h=HEADINGS if en else HEADINGS_ID
    n=analysis['n']; convert,bib,_,_,_=citation_plan(m,v,refs,include_findings=n>0)
    if not ready: d.add_paragraph('WORKING DRAFT — NOT READY FOR SUBMISSION' if en else 'DRAF KERJA — BELUM SIAP SUBMIT',style='Subtitle')
    d.add_heading(convert(m.get('title','') or '[Title not supplied]'),0)
    if not blind: add_text(d,m.get('authors','') or '[Authors not supplied]')
    d.add_heading('Abstract' if en else 'Abstrak',1); add_text(d,convert(abstract_text(m,v)) or '[Not supplied]')
    d.add_paragraph(('Keywords: ' if en else 'Kata kunci: ')+convert(m.get('keywords','')))
    for k in p['order']:
        d.add_heading(p['heading_overrides'].get(k) or h[k],1)
        if k=='findings':
            for x in analysis['narrative']: add_text(d,x)
            if n:
                add_table(d,'Sentiment by case' if en else 'Sentimen antarkasus',analysis['cross'])
                add_table(d,'Human-coded themes' if en else 'Tema pengodean manusia',analysis['themes'])
                add_table(d,'Model evaluation' if en else 'Evaluasi model',analysis['metrics'])
            else:
                d.add_paragraph('[Author findings omitted: no validated publication scope.]' if en else '[Temuan peneliti tidak ditampilkan: cakupan publikasi kosong.]'); continue
        val=m.get(k,''); add_text(d,convert(val) or '[Not supplied / belum diisi]')
    if e.get('credit') and not blind: d.add_heading('Author contributions',1); add_text(d,e['credit'])
    if e.get('competing'): d.add_heading('Competing interests',1); add_text(d,e['competing'])
    d.add_heading('References' if en else 'Daftar pustaka',1)
    for s in bib:
        para=d.add_paragraph(s); para.paragraph_format.left_indent=Inches(.25); para.paragraph_format.first_line_indent=Inches(-.25)
    if not bib: d.add_paragraph('[No cited verified references / belum ada referensi terverifikasi yang disitasi dengan ID]')
    return doc_bytes(d)

def auxiliary(p,title,body):
    p=copy.deepcopy(p); p['columns']=1; d=make_doc(p); d.add_heading(title,0); add_text(d,body); return doc_bytes(d)

def package(m,v,refs,analysis,report,ready=False):
    p=v['profile']; e=v['extras']; b=io.BytesIO(); prefix='' if ready else 'DRAFT_'
    # A blind manuscript omits the author block; manual inspection still mandatory.
    with zipfile.ZipFile(b,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr(prefix+'Manuscript.docx',render(m,v,refs,analysis,blind=p['blind'],ready=ready))
        z.writestr('Title_Page.docx',auxiliary(p,'Title page',m.get('title','')+'\n'+m.get('authors','')+'\n'+e.get('corresponding','')))
        if p['cover'] or e.get('cover'): z.writestr(prefix+'Cover_Letter.docx',auxiliary(p,'Cover letter',e.get('cover') or '[Not supplied]'))
        if p['highlights'] or e.get('highlights'): z.writestr(prefix+'Highlights.docx',auxiliary(p,'Highlights',e.get('highlights') or '[Not supplied]'))
        z.writestr('Declarations.docx',auxiliary(p,'Research declarations','\n\n'.join(k+': '+m.get(k,'[Not supplied]') for k in ('ethics','availability','funding','ai_disclosure'))+'\nCRediT: '+e.get('credit','')+'\nCompeting interests: '+e.get('competing','')))
        z.writestr('Review_Report.txt','\n'.join(x['Status']+' | '+x['Pemeriksaan']+' | '+x['Detail'] for x in report))
        z.writestr('READ_ME.txt','Manual review required. No automatic submission. Check journal guidelines, anonymization (including self-citations, acknowledgements, metadata, institutions), data permissions and retention. No raw comments automatically exported. Drafts may contain unverified author-entered assertions. Do not submit a draft.\nJournal: '+p['journal']+'\nGuidelines: '+p['url'])
        _,bib,_,_,_=citation_plan(m,v,refs,include_findings=analysis['n']>0); z.writestr('References.txt','\n'.join(bib))
        if analysis['n'] and analysis.get('figure'): z.writestr('Figure_1.png',analysis['figure'])
    return b.getvalue()


def page(st,page_id,d,pub,api):
    """App entry. api contains only existing v8 pure functions + authorized save callback."""
    if st.session_state.cloud_role!='owner': st.error('Hanya pemilik proyek.'); st.stop()
    v=copy.deepcopy(st.session_state.get('journal9',defaults())); p=v['profile']; m=copy.deepcopy(pub['manuscript']); refs=pub['references']; e=v['extras']; ver=st.session_state.version
    def save(new=v,manuscript=None,action='Metadata jurnal diperbarui'):
        api['save'](new,manuscript,action)
    def show_report(report):
        st.dataframe(report,hide_index=True,use_container_width=True)
        st.caption('Pemeriksaan berbasis aturan dan konfirmasi peneliti; bukan review sejawat atau probabilitas diterima.')
    if page_id==13:
        st.header('Journal Finder'); st.caption('Katalog lokal: pencocokan kata kunci, bukan rekomendasi berdasarkan indeks atau metrik mutakhir. Kuartil, APC dan kebijakan harus diperiksa sendiri.')
        query=st.text_input('Topik penelitian',m.get('keywords','') or 'consumer behavior food marketing'); rows=find_journals(query); st.dataframe(rows,hide_index=True,use_container_width=True)
        names=[r['Jurnal'] for r in rows]; selected=st.selectbox('Pilih kandidat',names); row=next(r for r in rows if r['Jurnal']==selected); st.link_button('Buka situs jurnal',row['Situs'])
        if st.button('Gunakan jurnal ini'):
            old=p.copy(); v['profile']=default_profile(); v['profile'].update(journal=selected,publisher=row['Penerbit'],url=row['Situs'],language=old['language']); save(action='Jurnal tujuan diganti; profil aturan direset')
        st.markdown('Verifikasi indeks: [Scopus Sources](https://www.scopus.com/sources) · [SINTA](https://sinta.kemdiktisaintek.go.id/)')
        return
    if page_id==14:
        st.header('Journal Template Intelligence'); st.info('DOCX: salin pengaturan halaman, font Normal/Heading/Title dan kolom pertama. Dokumen keluaran dibuat bersih, bukan salinan layout kompleks, logo, header, atau objek template. PDF/TXT/DOCX pedoman: ekstraksi teks, bukan verifikasi otomatis.')
        temp=st.file_uploader('Template resmi Word (.docx, maksimal 5 MB)',type=['docx'],key='v9_template')
        if temp and st.button('Impor pengaturan template'):
            try:
                v['profile'],_=template_profile(temp.getvalue(),temp.name,p); save(action='Impor pengaturan template DOCX')
            except Exception as ex: st.error('Template tidak dapat dibaca atau format di luar batas. '+str(ex)[:150])
        guide=st.file_uploader('Author Guidelines PDF/DOCX/TXT (maks. 5 MB; PDF 50 halaman pertama)',type=['pdf','docx','txt'],key='v9_guide')
        if guide and st.button('Ekstrak teks pedoman'):
            try: p.update(guideline_text=extract_guidelines(guide.getvalue(),guide.name),guideline_name=guide.name[:200],verified=False,verified_date=''); save(action='Ekstraksi pedoman')
            except Exception: st.error('Pedoman gagal dibaca. Gunakan PDF teks atau tempel teks manual.')
        with st.form('v9_profile_'+str(ver)):
            q=copy.deepcopy(p)
            for k,label in [('journal','Nama jurnal (custom diperbolehkan)'),('publisher','Penerbit'),('url','URL pedoman resmi')]: q[k]=st.text_input(label,p[k],max_chars=1000)
            q['guideline_text']=st.text_area('Teks pedoman (boleh ditempel langsung)',p['guideline_text'],height=180,max_chars=80000)
            q['language']=st.selectbox('Bahasa output label otomatis',['English','Indonesia'],index=['English','Indonesia'].index(p['language']))
            q['citation']=st.selectbox('Gaya sitasi pendekatan (periksa manual)',['APA-like','Harvard-like','Numeric'],index=['APA-like','Harvard-like','Numeric'].index(p['citation']))
            q['abstract_type']=st.selectbox('Jenis abstrak',['Unstructured','Structured'],index=['Unstructured','Structured'].index(p['abstract_type']))
            for k,l in [('abstract_max','Batas kata abstrak'),('word_max','Batas kata naskah'),('keywords_max','Maksimum kata kunci')]: q[k]=st.number_input(l,0,1000000,int(p[k]),help='0 = belum ditetapkan')
            for k,l in [('blind','Naskah anonim'),('cover','Cover letter diperlukan'),('highlights','Highlights diperlukan')]: q[k]=st.checkbox(l,value=p[k])
            q['font']=st.text_input('Font',p['font'],max_chars=100)
            for k,l in [('size','Font isi (pt)'),('heading_size','Font heading (pt)'),('title_size','Font judul (pt)')]: q[k]=st.number_input(l,6.0,32.0,float(p[k]),step=.5)
            for k,l in [('margin_top','Margin atas (inci)'),('margin_bottom','Margin bawah (inci)'),('margin_left','Margin kiri (inci)'),('margin_right','Margin kanan (inci)')]: q[k]=st.number_input(l,.2,3.,float(p[k]),step=.05)
            q['page_width']=st.number_input('Lebar halaman (inci)',5.,15.,float(p['page_width'])); q['page_height']=st.number_input('Tinggi halaman (inci)',7.,20.,float(p['page_height'])); q['spacing']=st.number_input('Spasi',1.,3.,float(p['spacing']),step=.1); q['columns']=st.selectbox('Kolom',[1,2],index=int(p['columns'])-1)
            q['order']=st.multiselect('Bagian dan urutan ekspor (pilih ulang untuk mengurutkan)',BODY,default=p['order']); q['required']=st.multiselect('Bagian wajib jurnal',BODY,default=p['required']);
            overrides={}
            for k in q['order']:
                x=st.text_input('Nama heading '+k,p['heading_overrides'].get(k,''),key='v9heading_'+k+'_'+str(ver),max_chars=150)
                if x.strip(): overrides[k]=x
            q['heading_overrides']=overrides
            confirm=st.checkbox('Saya telah memeriksa pedoman resmi dan mengonfirmasi SEMUA pengaturan di atas (termasuk perubahan baru).',value=False)
            if st.form_submit_button('Simpan profil jurnal'):
                try: q.update(verified=confirm,verified_date=datetime.now(timezone.utc).date().isoformat() if confirm else ''); v['profile']=profile_decode(q); save(action='Profil jurnal disimpan')
                except ValueError as ex: st.error(str(ex))
        st.caption('Template terakhir: '+(p['template_name'] or 'belum ada')+' · Konfirmasi: '+(p['verified_date'] or 'belum'))
        st.subheader('Author Guidelines Analyzer — kandidat ketentuan dari teks')
        for line in guideline_hints(p['guideline_text']): st.write(line)
        if not p['guideline_text']: st.info('Unggah atau tempel pedoman terlebih dahulu.')
        return
    scope=api['scope'](d,pub); themes,ev=api['themes'](scope); metrics,_,_=api['metrics'](scope); cross=api['cross'](scope)
    n=len(scope); en=p['language']=='English'
    # Translate only fixed labels; never translate author-entered theme names.
    if en:
        names={'Positif':'Positive','Negatif':'Negative','Netral':'Neutral','Campuran':'Mixed','Tidak dapat ditentukan':'Undetermined'}
        narrative=[f'Stored corpus: {len(d)} comments. Unit of analysis: comment, not unique user.']
        if n:
            counts=scope.Label_Referensi.value_counts(); narrative += [f'Validated publication scope: {n} comments across {scope.Kasus.nunique()} cases.', 'Human reference labels: '+', '.join(f'{names[k]}: {counts.get(k,0)} ({counts.get(k,0)/n:.1%})' for k in names)+'.', f'Human theme coding covers {ev.ID.nunique()} of {n} comments. Frequency is not thematic importance.', 'Descriptive results do not establish actual visits, causality or population generalizability.']
        else: narrative+=['No validated publication scope. No final empirical findings are generated.']
        cross=cross.rename(columns={**names,'Kasus':'Case'}); themes=themes.rename(columns={'Tema':'Theme','N_komentar':'Comments','N_kasus':'Cases','Kategori':'Category','Contoh_ID':'Example codes'}); metrics=metrics.rename(columns={'Metrik':'Metric','Nilai':'Value'})
    else: narrative=api['narrative'](d,pub)
    analysis=dict(n=n,narrative=narrative,cross=cross,themes=themes,metrics=metrics,figure=None)
    report=checks(m,v,refs,n,pub['checks_basis']==api['readiness'](d,pub) and len(pub['checks'])==api['check_count'],api['mapping'](pub,scope) or ([] if pub['theory'] else ['Pemetaan teori belum ada']),len(themes))
    if page_id==15:
        st.header('Automatic Manuscript Builder'); st.caption('Menggabungkan draf menu 12, data tervalidasi dan profil jurnal. Tidak menulis literatur atau interpretasi baru tanpa input peneliti. Simpan untuk memakai hasil edit di menu 12 juga.')
        for text in narrative: st.write(text)
        with st.expander('Impor draf artikel Word yang sudah ada (opsional)'):
            st.caption('Hanya paragraf dengan heading yang dikenali; tabel, gambar, footnote, daftar pustaka dan bagian tak dikenali tidak diimpor otomatis. Periksa pratinjau. Draf lama dicadangkan sebagai versi manual.')
            upload=st.file_uploader('Draf .docx, maksimal 5 MB',type=['docx'],key='v9_draft_import')
            if upload:
                try:
                    imported,unassigned=import_manuscript(upload.getvalue()); st.write('Bagian terdeteksi:',list(imported)); st.dataframe([{'Bagian':k,'Pratinjau':val[:240]} for k,val in imported.items()],hide_index=True)
                    if unassigned: st.write('Contoh teks yang tidak dipetakan:',unassigned[:10])
                    confirm=st.checkbox('Saya diizinkan menggunakan isi dokumen ini dan setuju mengganti bagian yang terdeteksi.',key='v9_import_confirm_'+str(ver))
                    if st.button('Impor bagian terdeteksi',disabled=not(imported and confirm)):
                        old=add_history(v,m,'Sebelum impor DOCX',basis(api['records'](d),pub)); save(old,{**m,**imported},'Impor draf DOCX; verifikasi ulang isi dan sitasi')
                except Exception: st.error('Dokumen gagal dibaca. Gunakan DOCX teks dengan heading yang jelas.')
        with st.form('v9_manuscript_'+str(ver)):
            new={k:st.text_area(api['fields'][k],m.get(k,''),max_chars=50000,height=90 if k in ('title','authors','keywords') else 130) for k in FIELDS}
            for k,l in STRUCT: e[k]=st.text_area(l+' (structured abstract)',e.get(k,''),max_chars=50000)
            for k,l in [('gap_evidence','Research gap: perbandingan literatur terverifikasi'),('novelty_evidence','Novelty: argumentasi dan bukti'),('credit','CRediT / kontribusi penulis'),('competing','Konflik kepentingan'),('corresponding','Corresponding author')]: e[k]=st.text_area(l,e.get(k,''),max_chars=50000)
            if st.form_submit_button('Simpan naskah ke proyek'): save(manuscript=new,action='Draf artikel v9 disimpan')
        if not n: st.warning('Cakupan publikasi kosong; bagian temuan peneliti tidak diekspor sebagai hasil pada manuskrip v9.')
        return
    if page_id==16:
        st.header('Scientific Reference & Citation'); st.caption('Referensi bersumber dari menu 11. Gunakan token [[R01]] pada naskah. Hanya referensi terverifikasi dan lengkap dipakai. Format APA/Harvard/Numeric bersifat pendekatan, bukan mesin CSL lengkap.')
        st.dataframe(refs,hide_index=True,use_container_width=True)
        with st.form('v9_citations_'+str(ver)):
            labels={}
            for r in refs:
                rid=r['ID_Referensi']; labels[rid]=st.text_input(rid+' — label sitasi (mis. Kozinets, 2020)',v['citation_labels'].get(rid,''),max_chars=200,key='v9cite_'+rid+'_'+str(ver))
            if st.form_submit_button('Simpan label sitasi'): v['citation_labels']=labels; save(action='Label sitasi disimpan')
        _,bib,unknown,missing,unused=citation_plan(m,v,refs,include_findings=n>0)
        st.write('ID belum sah:',unknown,'Label kosong/duplikat:',missing,'Referensi terverifikasi belum disitasi:',unused)
        for s in bib: st.write(s)
        st.caption('Tambahkan atau ubah metadata bibliografi dan verifikasi sumber di menu 11. Sitasi yang diketik bebas tidak dapat diperiksa otomatis.')
        return
    if page_id==17:
        st.header('Scientific Reviewer Engine'); show_report(report)
        st.subheader('Audit substansi (wajib diperiksa manusia)'); st.write('Kebaruan, kecukupan data, kualitas interpretasi, validitas konstruk, ketepatan referensi, dan kelayakan etik tidak dapat disimpulkan dari kelengkapan formulir.')
        st.write('Catatan gap:',e.get('gap_evidence') or 'Belum tersedia'); st.write('Catatan novelty:',e.get('novelty_evidence') or 'Belum tersedia'); return
    if page_id==18:
        st.header('Journal Compliance Checker'); show_report(report)
        st.write('Profil:',p['journal'] or 'Belum dipilih','Format:',p['font'],p['size'],'pt; kolom:',p['columns']); st.caption('Pemeriksaan panjang kata tidak menghitung seluruh tabel/referensi. Struktur abstrak, gaya sitasi, dan kebijakan jurnal perlu verifikasi manual.')
        return
    if page_id==19:
        st.header('Submission Package'); st.caption('Ekspor DOCX dan ZIP. Tidak melakukan submission otomatis. Identitas pada naskah blind hanya dihilangkan dari blok penulis; periksa teks, self-citation, pendanaan dan metadata secara manual.')
        if st.button('Buat kerangka cover letter otomatis (draf)'):
            title=m.get('title') or '[Article title]'; journal=p['journal'] or '[Journal name]'
            v['extras']['cover']=f'Dear Editor,\n\nPlease consider the manuscript entitled “{title}” for {journal}.\n\n[Explain journal fit, validated findings, and original contribution.]\n\n[Confirm authorship, competing interests, originality and exclusive submission only if true.]\n\nSincerely,\n{m.get("authors", "[Author]")}'
            save(action='Kerangka cover letter dibuat; peneliti wajib melengkapi')
        with st.form('v9_submission_'+str(ver)):
            e['cover']=st.text_area('Cover letter (ditulis / diperiksa peneliti)',e.get('cover',''),max_chars=50000); e['highlights']=st.text_area('Highlights (satu poin per baris)',e.get('highlights',''),max_chars=50000)
            if st.form_submit_button('Simpan dokumen tambahan'): save(action='Dokumen tambahan disimpan')
        show_report(report)
        all_ok=all(x['Status']=='OK' for x in report); st.caption('Label draf dipertahankan sampai seluruh pemeriksaan internal lulus dan peneliti mengonfirmasi pemeriksaan akhir.')
        export_basis=hashlib.sha256(json.dumps([basis(api['records'](d),pub),m,p,e,v['citation_labels'],report],sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:16]
        allowed=st.checkbox('Izin ekspor, anonimitas, dan retensi telah diperiksa; naskah tidak mengandung informasi terlarang.',key='v9allow_'+export_basis); review=st.checkbox('Saya telah memeriksa substansi, semua teks, format jurnal, dan blind review secara manual.',key='v9final_'+export_basis,disabled=not all_ok)
        if allowed:
            ready=all_ok and review
            try:
                if n: analysis['figure']=api['figure'](api['cross'](scope),'cross')
                prefix='' if ready else 'DRAFT_'; st.download_button('Unduh Word sesuai profil jurnal',render(m,v,refs,analysis,blind=p['blind'],ready=ready),prefix+'NetnoSense_9_Manuscript.docx',mime='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
                st.download_button('Unduh paket ZIP',package(m,v,refs,analysis,report,ready),prefix+'NetnoSense_9_Submission.zip',mime='application/zip')
            except Exception: st.error('Ekspor gagal. Periksa isi teks, ukuran halaman, dan library. Data proyek tidak dihapus.')
        return
    if page_id==20:
        st.header('Manuscript Version History'); st.caption('Maksimal 20 snapshot manual dalam proyek. Riwayat menyimpan teks, profil dan label sitasi, bukan salinan komentar. Retensi proyek tetap berlaku; versi lama terhapus ketika batas 20 terlampaui.')
        stamp=basis(api['records'](d),pub); note=st.text_input('Catatan revisi',max_chars=500)
        if st.button('Simpan versi naskah saat ini'): save(add_history(v,m,note,stamp),action='Versi naskah dibuat')
        if v['history']:
            st.dataframe([{k:r[k] for k in ('id','utc','note')} for r in v['history']],hide_index=True); ids={r['id']:r for r in v['history']}; sel=st.selectbox('Versi',list(ids)); r=ids[sel]; st.write('Judul:',r['manuscript'].get('title','')); st.write('Kesesuaian basis data saat ini:',r['basis']==stamp)
            confirm=st.checkbox('Saya setuju mengganti draf dan profil aktif, tanpa mengubah komentar, coder, atau referensi.')
            if st.button('Pulihkan versi naskah',disabled=not confirm):
                v=add_history(v,m,'Sebelum pemulihan '+sel,stamp); v.update(profile=r['profile'],extras=r['extras'],citation_labels=r['citation_labels']); v['profile']['verified']=False; v['profile']['verified_date']=''; save(v,r['manuscript'],'Pulihkan versi naskah; periksa ulang referensi dan pedoman')
        return
