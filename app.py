import io, re
import requests
from urllib.parse import urlparse, parse_qs
from datetime import datetime, timezone
from wordcloud import WordCloud
from collections import Counter
import pandas as pd
import plotly.express as px
import streamlit as st
from transformers import pipeline


MODEL='w11wo/indonesian-roberta-base-sentiment-classifier'
LABELS=['Positif','Negatif','Netral','Campuran','Tidak dapat ditentukan']
PENDING='Belum divalidasi'
@st.cache_resource
def load_model():
    return pipeline('sentiment-analysis',model=MODEL,device=-1)

ASPECTS={'Makanan':r'\b(makanan(?:nya)?|rasa(?:nya)?|enak|hambar|porsi(?:nya)?|tekstur|menu)\b','Harga':r'\b(harga(?:nya)?|mahal|murah|worth|terjangkau)\b','Pelayanan':r'\b(pelayanan(?:nya)?|pelayan|ramah|antri|antre|lambat)\b','Suasana':r'\b(tempat|suasana|bersih|kotor|nyaman)\b','Kreator':r'\b(vlogger|reviewer|influencer|kreator|endorse|review)\b'}
BEHAVIOR={'Curiosity':r'penasaran|membuktikan|bukti sendiri','Avoidance intention':r'(tidak|nggak|gak|ga)\s+(mau|ingin)\s+(datang|coba|beli|makan)|kapok','Visit intention':r'(mau|ingin|pengen|akan|pasti)\s+(datang|coba|mampir|berkunjung|mencoba)','Purchase intention':r'(mau|ingin|pengen)\s+beli','Reviewer skepticism':r'berlebihan|tidak objektif|nggak objektif|bohong|dibayar','FOMO':r'fomo|takut ketinggalan','Reported experience':r'sudah pernah|pernah makan|sudah mencoba'}
EMOTIONS={'Penasaran':r'penasaran|ingin tahu','Kecewa':r'kecewa|menyesal|zonk','Marah':r'marah|kesal|geram','Senang':r'senang|bahagia|puas','Simpati':r'kasihan|prihatin','Khawatir':r'takut|khawatir'}
BEHAVIOR.update({'Restaurant defense':r'padahal enak|tidak seburuk|nggak seburuk','Recommendation':r'rekomendasi|merekomendasikan','Verification motivation':r'membuktikan|bukti sendiri|buktikan'})

def redact(text):
    text=re.sub(r'https?://\S+', '[URL]', str(text))
    text=re.sub(r'[\w.+-]+@[\w.-]+\.[a-zA-Z]{2,}', '[EMAIL]', text)
    text=re.sub(r'(?<!\w)@[\w.]+', '[USER]', text)
    return re.sub(r'(?<!\w)(?:\+?\d[ .()-]?){8,15}(?!\w)', '[NUMBER]', text)

def video_id(value, platform):
    value=value.strip()
    if platform=='YouTube' and re.fullmatch(r'[A-Za-z0-9_-]{11}',value): return value
    if platform=='TikTok' and re.fullmatch(r'\d{10,25}',value): return value
    u=urlparse(value); host=(u.hostname or '').lower()
    if platform=='YouTube':
        v=parse_qs(u.query).get('v',[''])[0] if host in ('youtube.com','www.youtube.com','m.youtube.com') else ''
        if host=='youtu.be': v=u.path.strip('/').split('/')[0]
        if host in ('youtube.com','www.youtube.com','m.youtube.com') and u.path.startswith(('/shorts/','/live/','/embed/')): v=u.path.split('/')[2]
        if re.fullmatch(r'[A-Za-z0-9_-]{11}',v): return v
    if platform=='TikTok' and host in ('tiktok.com','www.tiktok.com','m.tiktok.com'):
        m=re.search(r'/video/(\d{10,25})(?:/|$)',u.path)
        if m: return m.group(1)
    raise ValueError('URL atau ID video tidak valid. Gunakan URL lengkap, bukan tautan pendek TikTok.')

def collect(platform, url, credential, limit, order='time'):
    vid=video_id(url,platform); rows=[]; seen=set(); cursor=0; page=None
    stamp=datetime.now(timezone.utc).isoformat(); status='Target tercapai'
    for _ in range(100):
        try:
            if platform=='YouTube':
                params={'part':'snippet','videoId':vid,'maxResults':min(100,limit-len(rows)),'textFormat':'plainText','order':order,'key':credential}
                if page: params['pageToken']=page
                r=requests.get('https://www.googleapis.com/youtube/v3/commentThreads',params=params,timeout=25)
                if not r.ok: raise ValueError(f'YouTube HTTP {r.status_code}: periksa API key, kuota, dan akses komentar.')
                obj=r.json(); items=[(x['snippet']['topLevelComment']['id'],x['snippet']['topLevelComment']['snippet']) for x in obj.get('items',[])]
                next_page=obj.get('nextPageToken'); more=bool(next_page) and next_page!=page; page=next_page
            else:
                r=requests.post('https://open.tiktokapis.com/v2/research/video/comment/list/',params={'fields':'id,text,like_count,create_time,reply_count'},headers={'Authorization':'Bearer '+credential},json={'video_id':int(vid),'max_count':min(100,limit-len(rows)),'cursor':cursor},timeout=25)
                if not r.ok: raise ValueError(f'TikTok HTTP {r.status_code}: periksa token dan izin Research API.')
                obj=r.json()
                if obj.get('error',{}).get('code') not in (None,'ok'): raise ValueError('TikTok menolak permintaan API. Periksa izin dan token.')
                data=obj.get('data',{}); items=[(str(x['id']),x) for x in data.get('comments',[])]; nxt=data.get('cursor',cursor); more=bool(data.get('has_more')) and nxt!=cursor; cursor=nxt
            for ident,x in items:
                if ident in seen: continue
                seen.add(ident)
                text=redact(x.get('textOriginal',x.get('textDisplay',x.get('text',''))))
                rows.append({'Komentar':text,'Video_ID':vid,'Likes':x.get('likeCount',x.get('like_count',0)),'Waktu_komentar':str(x.get('publishedAt',x.get('create_time',''))),'Waktu_koleksi_UTC':stamp,'Metode_koleksi':platform+' API / komentar utama / '+(order if platform=='YouTube' else 'urutan API')})
                if len(rows)>=limit: break
            if len(rows)>=limit: break
            if not more: status='Halaman API habis sebelum target'; break
        except (requests.RequestException,ValueError,KeyError) as e:
            status=str(e) if isinstance(e,ValueError) and not isinstance(e,requests.RequestException) else 'Permintaan API gagal; periksa koneksi atau respons.'
            if not rows: raise ValueError(status) from None
            break
    else: status='Batas halaman tercapai'
    return pd.DataFrame(rows,columns=['Komentar','Video_ID','Likes','Waktu_komentar','Waktu_koleksi_UTC','Metode_koleksi']),status

def tags(text,patterns):
    return '; '.join(k for k,v in patterns.items() if re.search(v,text,re.I)) or 'Belum teridentifikasi'

def safe_cell(v):
    return "'"+v if isinstance(v,str) and v.lstrip().startswith(('=','+','-','@')) else v


# V4: all automatic codes are suggestions, not validated findings.
import json, hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score

COLORS={'Positif':'#16834b','Negatif':'#c93838','Netral':'#94a3b8','Campuran':'#d97706','Tidak dapat ditentukan':'#64748b'}
UNKNOWN='Belum teridentifikasi'
REL=['Belum diperiksa','Relevan','Tidak relevan','Spam/promosi','Ambigu']
NORMAL={'gk':'tidak','ga':'tidak','gak':'tidak','nggak':'tidak','ngga':'tidak','enggak':'tidak','tdk':'tidak','bgt':'banget','bnget':'banget','bgtt':'banget','udah':'sudah','sdh':'sudah','dlu':'dulu','dl':'dulu','skrg':'sekarang','skrng':'sekarang','pengen':'ingin','pingin':'ingin','cobain':'coba','nyoba':'coba','nyobain':'coba','mencoba':'coba','dateng':'datang','dtg':'datang','antri':'antre','antrian':'antrean','makanannya':'makanan','pelayanannya':'pelayanan','harganya':'harga','rasanya':'rasa','tempatnya':'tempat','rekomendasiin':'rekomendasikan','enakkk':'enak','jg':'juga','jgn':'jangan','yg':'yang','krn':'karena','klo':'kalau','kalo':'kalau'}
STOP=set('yang dan di ke dari ini itu saya aku nya untuk dengan ada juga tapi kalau sudah sangat jadi koh bang kak min mas mbak si nih dong deh lah kok ya aja pada dalam tersebut adalah karena atau sebagai'.split())
ASPECTS.update({'Makanan':r'\b(makanan|rasa|enak|hambar|porsi|tekstur|menu|bebek|ayam|nasi|sambal|sambel|lezat|gurih|asin|pedas|mentah|basi)\b','Pelayanan':r'\b(pelayanan|pelayan|ramah|antre|antrean|lambat|cepat|nunggu|menunggu)\b','Konten/iklan':r'\b(iklan|video|editing|durasi|konten|audio|sponsor)\b','Kreator':r'\b(vlogger|reviewer|influencer|kreator|endorse|review|ulasan|host)\b','Restoran umum':r'\b(restoran|warung|kedai|rumah makan|tempat makan)\b','Komunitas':r'\b(komentar|komentator|netizen|penonton|audiens)\b'})
BEHAVIOR.update({'Reported experience':r'\b(sudah (pernah |\w+ )?(coba|makan|datang)|pernah (makan|coba|datang)|langganan|dulu (sering )?makan|sering makan|habis makan|baru (makan|coba))\b','Repeat patronage':r'\b(langganan|sering (makan|datang|beli)|datang lagi|beli lagi)\b','Visit intention':r'\b(?:mau|ingin|akan|pasti|besok|nanti|kapan kapan)\s+(?:\w+\s+){0,2}(?:datang|coba|mampir|berkunjung|makan)\b','Avoidance intention':r'\b(?:tidak|jangan|ogah|malas)\s+(?:\w+\s+){0,2}(?:datang|coba|beli|makan|mampir)|\bkapok\b','Curiosity':r'penasaran|ingin tahu|bukti sendiri|membuktikan|emang seenak|seburuk itu','Recommendation':r'\b(rekomendasi|rekomendasikan|recommended|wajib coba|harus coba|coba deh|coba juga)\b','Queue observation':r'\b(antre|antrean|ramai|penuh|habis|sold out)\b','Reviewer skepticism':r'berlebihan|tidak objektif|bohong|dibayar|settingan|lebay|tidak jujur|endorse','Verification motivation':r'membuktikan|bukti sendiri|buktikan|coba sendiri','Purchase intention':r'\b(mau|ingin|akan)\s+beli\b'})
EMOTIONS.update({'Senang':r'\b(senang|bahagia|puas|suka|mantap|joss|jos|mantul)\b','Kecewa':r'\b(kecewa|menyesal|zonk|kapok)\b','Simpati':r'kasihan|prihatin|semoga','Khawatir':r'khawatir|takut|waswas'})

def normalize(t):
    t=str(t).lower().replace('’',"'")
    t=re.sub(r'([a-z])\1{2,}',r'\1',t)
    return re.sub(r'\b\w+\b',lambda m:NORMAL.get(m.group(),m.group()),t)

def behavior(t):
    found=[k for k,v in BEHAVIOR.items() if re.search(v,t,re.I)]
    # Negation handling is conservative: no positive-intent tag if all matching phrases are negated.
    for key in ('Visit intention','Purchase intention'):
        if key in found:
            clean=re.sub(r'\b(?:tidak|jangan|ogah|malas)\s+(?:\w+\s+){0,2}(?:datang|coba|beli|makan|mampir)\b',' ',t)
            if not re.search(BEHAVIOR[key],clean): found.remove(key)
    return '; '.join(found) or UNKNOWN

def relevance(t, extra):
    if re.search(r'\b(subscribe|promo|diskon|jualan|follow|hubungi|wa)\b',t) and ('[url]' in t or re.search(r'\b(?:klik|order|jual|hubungi)\b',t)): return 'Spam/promosi'
    if any(re.search(v,t) for v in ASPECTS.values()) or behavior(t)!=UNKNOWN or any(x in t for x in extra): return 'Relevan'
    return 'Ambigu'

def counts_of(series):
    return pd.DataFrame(Counter(k for x in series.fillna('') for k in str(x).split('; ') if k and k!=UNKNOWN).most_common(),columns=['Kode','Jumlah'])

def predict(model, texts):
    out=[]
    for i in range(0,len(texts),16):
        out.extend(model(texts[i:i+16],truncation=True,batch_size=4))
    mapping={'positive':'Positif','negative':'Negatif','neutral':'Netral'}
    return [(mapping.get(x['label'].lower(),'Tidak dapat ditentukan'),round(float(x['score']),4)) for x in out]

def analyze(raw, model, extra, project, case, source):
    d=raw.copy().dropna(subset=['Komentar']); d['Komentar']=d.Komentar.map(redact).str.strip(); d=d[d.Komentar!=''].reset_index(drop=True)
    if d.empty: raise ValueError('Tidak ada komentar yang dapat dianalisis.')
    if 'Kasus' not in d: d['Kasus']=case
    if 'Sumber' not in d: d['Sumber']=source
    d['Kasus']=d.Kasus.fillna(case).astype(str); d['Sumber']=d.Sumber.fillna(source).astype(str)
    d['Proyek']=project; d['ID']=[hashlib.sha256((str(datetime.now(timezone.utc))+str(i)).encode()).hexdigest()[:16] for i in range(len(d))]
    d['Normalisasi']=d.Komentar.map(normalize)
    pred=predict(model,d.Normalisasi.tolist()); d['Sentimen_AI']=[x[0] for x in pred]; d['Skor_Model']=[x[1] for x in pred]
    d['Relevansi_Saran']=d.Normalisasi.map(lambda t:relevance(t,extra)); d['Relevansi']='Belum diperiksa'
    d['Kode_Saran']=d.Normalisasi.map(behavior); d['Emosi_Saran']=d.Normalisasi.map(lambda t:tags(t,EMOTIONS)); d['Aspek_Saran']=d.Normalisasi.map(lambda t:tags(t,ASPECTS))
    clauses=[]
    for ident,t in zip(d.ID,d.Normalisasi):
        for clause in re.split(r'[.!?;]+|\b(?:tetapi|tapi|namun|sedangkan|walaupun)\b',t):
            if clause.strip(): clauses.append((ident,clause.strip()))
    cp=predict(model,[x[1] for x in clauses]); ar=[]; polar={}
    for (ident,clause),(sent,score) in zip(clauses,cp):
        polar.setdefault(ident,set()).add(sent)
        for a,p in ASPECTS.items():
            if re.search(p,clause): ar.append({'ID':ident,'Klausa_normalisasi':clause,'Aspek':a,'Sentimen_Saran':sent,'Skor_Model':score,'Validasi':PENDING})
    d['Sentimen_Konteks_Saran']=[ 'Campuran' if {'Positif','Negatif'}<=polar.get(i,set()) else s for i,s in zip(d.ID,d.Sentimen_AI)]
    for c in ['Coder_1','Coder_2','Sentimen_Restoran','Sentimen_Kreator','Sentimen_Konten','Sentimen_Komunitas']: d[c]=PENDING
    for c in ['Kode_Peneliti','Kategori_Peneliti','Tema_Peneliti','Catatan']: d[c]=''
    d['Model']=MODEL
    return d,pd.DataFrame(ar,columns=['ID','Klausa_normalisasi','Aspek','Sentimen_Saran','Skor_Model','Validasi'])

def ngrams(texts,n):
    cnt=Counter()
    for t in texts:
        tokens=re.findall(r'\b[^\W\d_]+\b',t)
        for i in range(len(tokens)-n+1):
            window=tokens[i:i+n]
            if all(w not in STOP for w in window): cnt[' '.join(window)]+=1
    return pd.DataFrame(cnt.most_common(50),columns=['Frasa','Jumlah'])

def summary_for(d):
    return d.groupby('Sentimen_Final').size().reindex(LABELS,fill_value=0).rename_axis('Sentimen').reset_index(name='Jumlah')

def evaluate(d):
    valid=d[d.Coder_1.isin(LABELS[:3]) & d.Sentimen_AI.isin(LABELS[:3])]; pairs=d[d.Coder_1.isin(LABELS)&d.Coder_2.isin(LABELS)]
    met=[]; cm=pd.DataFrame(); report=pd.DataFrame()
    if len(valid):
        y=valid.Coder_1; p=valid.Sentimen_AI
        met.append({'Metrik':'Accuracy (3 kelas)','Nilai':float((y==p).mean()),'N':len(valid)})
        cm=pd.DataFrame(confusion_matrix(y,p,labels=LABELS[:3]),index=LABELS[:3],columns=LABELS[:3]); report=pd.DataFrame(classification_report(y,p,labels=LABELS[:3],output_dict=True,zero_division=0)).T
    if len(pairs) and len(set(pairs.Coder_1)|set(pairs.Coder_2))>1:
        met.append({'Metrik':'Cohen Kappa','Nilai':float(cohen_kappa_score(pairs.Coder_1,pairs.Coder_2,labels=LABELS)),'N':len(pairs)})
    return pd.DataFrame(met,columns=['Metrik','Nilai','N']),cm,report

# Interface
st.set_page_config(page_title='NetnoSense PRO 4.0',page_icon='📊',layout='wide')
st.markdown("""<style>[data-testid='stSidebar']{background: #f0f4fa;} div[data-testid='stMetric']{border:1px solid #d7e0eb;border-radius:12px;padding:14px;} .block-container{padding-top:2rem;} </style>""",unsafe_allow_html=True)
st.title('NetnoSense PRO 4.0')
st.caption('RESEARCH INTELLIGENCE  /  Digital consumer behavior & netnography')
with st.sidebar:
    project=st.text_input('Nama penelitian','Food Vlogger Research')
    page=st.radio('Navigasi',['01 · Koleksi data','02 · Validasi & coding','03 · Dashboard','04 · Tema & bahasa','05 · Evaluasi','06 · Ekspor'])
    st.caption('Data hanya disimpan dalam sesi. Ekspor sebelum menutup aplikasi.')
    if st.button('Kosongkan proyek',disabled=not st.checkbox('Konfirmasi reset data sesi')):
        for k in ['data','aspects','raw','collection_log','version','batch_hashes','raw_mode']: st.session_state.pop(k,None)
        st.rerun()
for k,default in [('data',pd.DataFrame()),('aspects',pd.DataFrame()),('raw',pd.DataFrame()),('collection_log',[]),('version',0),('batch_hashes',set())]:
    if k not in st.session_state: st.session_state[k]=default

if page.startswith('01'):
    st.header('Koleksi data')
    st.warning('Gunakan data yang diizinkan dan telah dianonimkan. Redaksi otomatis tidak menjamin anonimitas. Jangan unggah informasi pribadi atau API key ke GitHub.')
    mode=st.radio('Metode input',['Input manual','Upload Excel/CSV','YouTube API','TikTok Research API'],horizontal=True)
    case=st.text_input('Kode kasus / awalan kode video','V01')
    raw=None; source='Lainnya'
    if mode in ['Input manual','Upload Excel/CSV']:
        source=st.selectbox('Sumber data',['TikTok','YouTube','Instagram','Google Reviews','X','Lainnya'])
        if mode=='Input manual': raw=pd.DataFrame({'Komentar':st.text_area('Satu komentar anonim per baris',height=180).splitlines()})
        else:
            f=st.file_uploader('Dataset anonim (Excel/CSV)',type=['xlsx','csv'])
            if f:
                try:
                    x=pd.read_csv(f) if f.name.lower().endswith('.csv') else pd.read_excel(f)
                    c=st.selectbox('Kolom komentar',x.columns,index=list(x.columns).index('Komentar') if 'Komentar' in x else 0)
                    raw=x[[c]].rename(columns={c:'Komentar'}).copy()
                    for col in ['Kasus','Sumber','Video_ID','Likes','Waktu_komentar','Waktu_koleksi_UTC','Metode_koleksi']:
                        if col in x and col!=c: raw[col]=x[col]
                except Exception: st.error('File tidak dapat dibaca. Periksa format dan nama kolom.')
    else:
        source='YouTube' if mode=='YouTube API' else 'TikTok'
        st.info('Sumber otomatis: '+source+'. Hanya komentar utama; akses TikTok membutuhkan persetujuan Research API.')
        urls=st.text_area('URL atau ID video, satu per baris (maksimal 10)')
        credential=st.text_input('API key YouTube' if source=='YouTube' else 'Access token TikTok',type='password')
        limit=st.number_input('Target komentar per video',1,2000,100)
        order=st.selectbox('Urutan YouTube',['time','relevance']) if source=='YouTube' else 'API'
        allowed=st.checkbox('Saya memiliki izin menggunakan API dan data ini untuk penelitian')
        if st.button('Ambil komentar',disabled=not allowed):
            st.session_state.raw=pd.DataFrame()
            inputs=list(dict.fromkeys(z.strip() for z in urls.splitlines() if z.strip()))
            if not credential or not inputs or len(inputs)>10: st.error('Isi kredensial dan 1–10 URL/ID.')
            else:
                batches=[]; log=[]; seen_v=set()
                with st.spinner('Mengambil komentar melalui API resmi...'):
                    for i,u in enumerate(inputs):
                        try:
                            vid=video_id(u,source)
                            if vid in seen_v: continue
                            seen_v.add(vid)
                            batch,status=collect(source,u,credential,int(limit),order)
                            batch['Kasus']=case if len(inputs)==1 else f'{case}-{i+1:02d}'; batch['Sumber']=source
                            batches.append(batch); log.append({'Platform':source,'Video_ID':vid,'Kasus':batch.Kasus.iloc[0] if len(batch) else case,'Target':int(limit),'Diperoleh':len(batch),'Status':status})
                        except ValueError as e: log.append({'Platform':source,'Video_ID':'','Kasus':case,'Target':int(limit),'Diperoleh':0,'Status':str(e)})
                st.session_state.raw=pd.concat(batches,ignore_index=True) if batches else pd.DataFrame(); st.session_state.collection_log.extend(log)
                st.session_state.raw_mode=mode
                st.dataframe(pd.DataFrame(log),hide_index=True)
        if st.session_state.get('raw_mode')==mode and not st.session_state.raw.empty:
            raw=st.session_state.raw.copy(); st.success(f'{len(raw)} komentar tersedia dari {source}.'); st.dataframe(raw.head(10),hide_index=True)
            st.download_button('Unduh koleksi CSV',raw.map(safe_cell).to_csv(index=False).encode('utf-8-sig'),'koleksi.csv')
    extra=[normalize(x.strip()) for x in st.text_input('Kata kunci relevansi tambahan (pisahkan koma)','').split(',') if x.strip()]
    dedup=st.checkbox('Hapus teks identik dalam kasus yang sama (opsional)',False)
    if st.button('Analisis & tambahkan ke proyek',type='primary'):
        if raw is None or raw.empty: st.warning('Masukkan atau ambil komentar terlebih dahulu.')
        else:
            try:
                if dedup: raw=raw.drop_duplicates(subset=['Komentar']+(['Kasus'] if 'Kasus' in raw else []))
                if not st.session_state.data.empty and project != st.session_state.data.Proyek.iloc[0]:
                    raise ValueError('Ekspor dan reset sebelum mengganti nama proyek.')
                fingerprint=hashlib.sha256((project+source+case+raw.fillna('').astype(str).drop(columns=['Waktu_koleksi_UTC'],errors='ignore').to_csv(index=False)).encode()).hexdigest()
                if fingerprint in st.session_state.batch_hashes: raise ValueError('Batch identik sudah ditambahkan.')
                with st.spinner('Model sedang memproses komentar dan klausa...'):
                    d,a=analyze(raw,load_model(),extra,project,case,source)
                st.session_state.data=pd.concat([st.session_state.data,d],ignore_index=True); st.session_state.aspects=pd.concat([st.session_state.aspects,a],ignore_index=True); st.session_state.version+=1; st.session_state.batch_hashes.add(fingerprint)
                st.success(f'{len(d)} komentar ditambahkan. Buka menu 02 atau 03. Jangan klik ulang untuk batch yang sama.')
            except ValueError as e: st.error(str(e))
            except Exception as e: st.error('Analisis gagal. Periksa format data atau log aplikasi. '+type(e).__name__)
    st.metric('Komentar dalam sesi proyek',len(st.session_state.data))
    if st.session_state.collection_log: st.dataframe(pd.DataFrame(st.session_state.collection_log),hide_index=True)
    st.stop()

if st.session_state.data.empty: st.info('Belum ada data. Buka menu 01 · Koleksi data.'); st.stop()
d=st.session_state.data.copy(); aspects=st.session_state.aspects.copy()
if page.startswith('02'):
    st.header('Validasi & pengodean')
    st.caption('Coder harus bekerja independen tanpa melihat prediksi atau label coder lain. Mode di bawah menyembunyikan kolom tersebut, tetapi bukan kontrol akses multiuser.')
    mode=st.radio('Mode kerja',['Peneliti / pengodean','Coder 1 (buta)','Coder 2 (buta)'],horizontal=True)
    if mode.startswith('Coder'):
        col='Coder_1' if mode.startswith('Coder 1') else 'Coder_2'; visible=['ID','Komentar','Kasus',col]; editable=[col]
    else:
        editable=['Relevansi','Sentimen_Restoran','Sentimen_Kreator','Sentimen_Konten','Sentimen_Komunitas','Kode_Peneliti','Kategori_Peneliti','Tema_Peneliti','Catatan']; visible=['ID','Komentar','Kasus','Sumber','Sentimen_AI','Sentimen_Konteks_Saran','Skor_Model','Relevansi_Saran','Aspek_Saran','Kode_Saran','Emosi_Saran']+editable
    config={c:st.column_config.SelectboxColumn(c,options=[PENDING]+LABELS) for c in editable if c.startswith(('Coder','Sentimen'))}; config['Relevansi']=st.column_config.SelectboxColumn('Relevansi',options=REL)
    with st.form('validation'):
        edited=st.data_editor(d[visible],disabled=[c for c in visible if c not in editable],column_config=config,hide_index=True,key=f'edit_{mode}_{st.session_state.version}')
        if st.form_submit_button('Simpan pengodean'):
            for c in editable: d[c]=edited[c].values
            st.session_state.data=d; st.session_state.version+=1; st.rerun()
    if mode=='Peneliti / pengodean' and not aspects.empty:
        st.subheader('Validasi aspek / klausa')
        st.caption('Aspek berbasis kamus, polaritas klausa dari model umum; bukan ABSA terlatih.')
        with st.form('aspect_form'):
            ae=st.data_editor(aspects,disabled=[c for c in aspects if c!='Validasi'],column_config={'Validasi':st.column_config.SelectboxColumn('Validasi',options=[PENDING]+LABELS)},hide_index=True,key=f'ae_{st.session_state.version}')
            if st.form_submit_button('Simpan validasi aspek'):
                st.session_state.aspects=ae; st.session_state.version+=1; st.rerun()
    st.stop()

# Common filtered data: exclude only human-confirmed irrelevant/spam by default.
d['Sentimen_Final']=d.Coder_1.where(d.Coder_1!=PENDING,d.Sentimen_Konteks_Saran)
filter_mode=st.radio('Cakupan analisis',['Semua kecuali eksklusi terkonfirmasi','Hanya relevan terkonfirmasi','Semua komentar'],horizontal=True)
if filter_mode.startswith('Semua kecuali'): scope=d[~d.Relevansi.isin(['Tidak relevan','Spam/promosi'])].copy()
elif filter_mode.startswith('Hanya'): scope=d[d.Relevansi=='Relevan'].copy()
else: scope=d.copy()
summary=summary_for(scope); cross=scope.groupby(['Kasus','Sentimen_Final']).size().reset_index(name='Jumlah'); cross['Persen']=100*cross.Jumlah/cross.groupby('Kasus').Jumlah.transform('sum') if len(cross) else pd.Series(dtype=float)
a=aspects[aspects.ID.isin(scope.ID)].copy() if not aspects.empty else aspects.copy()
if not a.empty: a['Sentimen_Final']=a.Validasi.where(a.Validasi!=PENDING,a.Sentimen_Saran)
behavior_counts=counts_of(scope.Kode_Saran); emotion_counts=counts_of(scope.Emosi_Saran)
freqs={n:ngrams(scope.Normalisasi,n) for n in (1,2,3)}
metrics,cm,report=evaluate(d)
if page.startswith('03'):
    st.header('Dashboard penelitian'); cols=st.columns(4)
    for col,label,val in zip(cols,['Komentar dalam cakupan','Kasus','Coder 1 terisi','Relevansi belum diperiksa'],[len(scope),scope.Kasus.nunique(),int((scope.Coder_1!=PENDING).sum()),int((scope.Relevansi=='Belum diperiksa').sum())]): col.metric(label,val)
    st.caption('Distribusi sementara: label Coder 1 jika tersedia, jika tidak saran kontekstual AI. Tidak membuktikan perubahan kunjungan aktual.')
    if len(scope):
        st.plotly_chart(px.pie(summary[summary.Jumlah>0],names='Sentimen',values='Jumlah',hole=.55,color='Sentimen',color_discrete_map=COLORS))
        st.plotly_chart(px.bar(cross,x='Kasus',y='Persen',color='Sentimen_Final',barmode='group',color_discrete_map=COLORS,hover_data=['Jumlah'],title='Perbandingan antarkasus (%)'))
    st.subheader('Relevansi komentar'); st.dataframe(d.groupby(['Relevansi','Relevansi_Saran']).size().reset_index(name='Jumlah'),hide_index=True)
    for title,table in [('Kandidat perilaku',behavior_counts),('Indikasi emosi',emotion_counts)]:
        st.subheader(title)
        if not table.empty: st.plotly_chart(px.bar(table,x='Jumlah',y='Kode',orientation='h'))
        else: st.info('Belum ada kecocokan kata kunci.')
    if not a.empty:
        ac=a.groupby(['Aspek','Sentimen_Final']).size().reset_index(name='Jumlah'); st.plotly_chart(px.bar(ac,x='Aspek',y='Jumlah',color='Sentimen_Final',barmode='group',color_discrete_map=COLORS,title='Polaritas klausa per aspek (eksperimental)'))
    st.subheader('Sentimen per objek — label manusia')
    obj=scope.melt(id_vars=['ID'],value_vars=['Sentimen_Restoran','Sentimen_Kreator','Sentimen_Konten','Sentimen_Komunitas'],var_name='Objek',value_name='Label'); obj=obj[obj.Label!=PENDING]
    if len(obj): st.plotly_chart(px.bar(obj.groupby(['Objek','Label']).size().reset_index(name='Jumlah'),x='Objek',y='Jumlah',color='Label',barmode='group',color_discrete_map=COLORS))
    else: st.info('Isi label objek pada menu 02.')
elif page.startswith('04'):
    st.header('Eksplorasi bahasa & tema')
    st.caption('Normalisasi memakai kamus terbatas, bukan stemming linguistik penuh. Kelompok TF-IDF adalah kandidat eksplorasi, bukan tema netnografi yang sudah teruji.')
    tabs=st.tabs(['Unigram','Bigram','Trigram'])
    for n,tab in enumerate(tabs,1):
        with tab: st.dataframe(freqs[n],hide_index=True)
    if len(freqs[1]):
        wc=WordCloud(width=1200,height=600,background_color='white',random_state=42).generate_from_frequencies(dict(zip(freqs[1].Frasa,freqs[1].Jumlah))); st.image(wc.to_array()); png=io.BytesIO(); wc.to_image().save(png,format='PNG'); st.download_button('Unduh word cloud PNG',png.getvalue(),'wordcloud.png')
    st.subheader('Kandidat kelompok topik (TF-IDF + KMeans)')
    k=st.slider('Jumlah kelompok',2,10,3)
    if st.button('Eksplorasi kelompok topik'):
        try:
            texts=scope.Normalisasi.tolist()
            if len(set(texts))<k: raise ValueError('Komentar unik kurang dari jumlah kelompok.')
            v=TfidfVectorizer(ngram_range=(1,2),stop_words=list(STOP),max_features=3000); X=v.fit_transform(texts); km=KMeans(n_clusters=k,random_state=42,n_init=10).fit(X); terms=v.get_feature_names_out(); out=[]
            for j in range(k):
                ids=[i for i,c in enumerate(km.labels_) if c==j]; top=[terms[i] for i in km.cluster_centers_[j].argsort()[-6:][::-1] if km.cluster_centers_[j][i]>0]; out.append({'Kelompok':j+1,'N':len(ids),'Kata_kunci':', '.join(top),'Contoh_ID':', '.join(scope.ID.iloc[ids[:5]])})
            table=pd.DataFrame(out); st.dataframe(table,hide_index=True); st.download_button('Unduh kandidat topik CSV',table.to_csv(index=False).encode('utf-8-sig'),'topik.csv')
        except ValueError as e: st.warning(str(e))
    st.subheader('Tema yang ditetapkan peneliti'); st.dataframe(counts_of(scope.Tema_Peneliti),hide_index=True)
elif page.startswith('05'):
    st.header('Evaluasi model & reliabilitas'); st.caption('Evaluasi model dasar tiga kelas, bukan saran campuran. Gunakan label independen. N evaluasi dapat berbeda dari N proyek; pasangan Kappa mencakup lima kelas.')
    if metrics.empty: st.info('Isi Coder 1 dan Coder 2 di menu 02.')
    else: st.dataframe(metrics,hide_index=True)
    if not cm.empty: st.subheader('Confusion matrix (baris: manusia, kolom: AI)'); st.dataframe(cm); st.dataframe(report)
else:
    st.header('Pusat ekspor'); st.caption('Ekspor mencakup dataset lengkap dan ringkasan sesuai cakupan yang dipilih. Pastikan data telah dianonimkan sesuai etika dan ketentuan API sebelum disimpan atau dibagikan.')
    tables={'Dataset':d,'Cakupan':scope,'Ringkasan':summary,'Antarkasus':cross,'Aspek':a,'Perilaku':behavior_counts,'Emosi':emotion_counts,'Unigram':freqs[1],'Bigram':freqs[2],'Trigram':freqs[3],'Evaluasi':metrics,'Confusion_matrix':cm.reset_index(),'Metrik_per_kelas':report.reset_index(),'Log_koleksi':pd.DataFrame(st.session_state.collection_log)}
    buf=io.BytesIO()
    with pd.ExcelWriter(buf,engine='openpyxl') as writer:
        for name,table in tables.items(): table.map(safe_cell).to_excel(writer,sheet_name=name,index=False)
    st.download_button('Unduh seluruh hasil Excel',buf.getvalue(),'NetnoSense_PRO_4.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    text='NETNOSENSE PRO 4.0 — LAPORAN DESKRIPTIF\n'+f'Proyek: {project}\nCakupan: {filter_mode}\nN: {len(scope)} / {len(d)}\n'+summary.to_string(index=False)+'\n\n'+cross.to_string(index=False)+'\n\nMetrik model:\n'+metrics.to_string(index=False)+'\n\nBatasan: prediksi dan pengodean otomatis bersifat sementara. Komentar tidak membuktikan perubahan kunjungan atau hubungan kausal.'
    st.download_button('Unduh laporan TXT',text.encode('utf-8'),'laporan_v4.txt'); st.text(text)
