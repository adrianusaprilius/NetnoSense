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

st.set_page_config(page_title='NetnoSense PRO',page_icon='📊',layout='wide')
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
                rows.append({'Komentar':text,'Likes':x.get('likeCount',x.get('like_count',0)),'Waktu_komentar':str(x.get('publishedAt',x.get('create_time',''))),'Waktu_koleksi_UTC':stamp,'Metode_koleksi':platform+' API / komentar utama / '+(order if platform=='YouTube' else 'urutan API')})
                if len(rows)>=limit: break
            if len(rows)>=limit: break
            if not more: status='Halaman API habis sebelum target'; break
        except (requests.RequestException,ValueError,KeyError) as e:
            status=str(e) if isinstance(e,ValueError) and not isinstance(e,requests.RequestException) else 'Permintaan API gagal; periksa koneksi atau respons.'
            if not rows: raise ValueError(status) from None
            break
    else: status='Batas halaman tercapai'
    return pd.DataFrame(rows,columns=['Komentar','Likes','Waktu_komentar','Waktu_koleksi_UTC','Metode_koleksi']),status

def tags(text,patterns):
    return '; '.join(k for k,v in patterns.items() if re.search(v,text,re.I)) or 'Belum teridentifikasi'

def safe_cell(v):
    return "'"+v if isinstance(v,str) and v.lstrip().startswith(('=','+','-','@')) else v

st.title('NetnoSense PRO')
st.caption('Digital Consumer Research Intelligence · v3.0')
st.warning('Gunakan data yang telah dianonimkan. Aplikasi cloud bukan tempat penyimpanan data pribadi. Prediksi AI dan kode berbasis aturan wajib diperiksa peneliti.')
with st.sidebar:
    st.header('Research Settings')
    project=st.text_input('Nama penelitian','Food Vlogger Research')
    source=st.selectbox('Sumber data',['TikTok','YouTube','Instagram','Google Reviews','X','Lainnya'])
    case=st.text_input('Kode kasus / video','V01')
    dedup=st.checkbox('Hapus duplikat dalam impor ini',False)
    st.caption('Model: '+MODEL)
mode=st.radio('Metode input',['Input manual','Upload Excel/CSV','YouTube API','TikTok Research API'],horizontal=True)
raw=None
if mode=='Input manual':
    s=st.text_area('Satu komentar anonim per baris',height=180)
    raw=pd.DataFrame({'Komentar':s.splitlines()})
elif mode=='Upload Excel/CSV':
    f=st.file_uploader('Dataset anonim',type=['xlsx','csv'])
    if f:
        try:
            data=pd.read_excel(f) if f.name.lower().endswith('.xlsx') else pd.read_csv(f)
            col=st.selectbox('Kolom komentar',data.columns)
            raw=data[[col]].rename(columns={col:'Komentar'}).copy()
            case_col=st.selectbox('Kolom kasus (opsional)', ['-- satu kasus --']+[c for c in data.columns if c!=col])
            if case_col!='-- satu kasus --': raw['Kasus']=data[case_col].fillna(case).astype(str)
        except Exception as e: st.error(str(e))
else:
    platform='YouTube' if mode=='YouTube API' else 'TikTok'
    st.info('Hanya komentar utama. Gunakan akses resmi dan data yang diizinkan. Redaksi otomatis tidak menjamin anonimitas.')
    url=st.text_input('URL lengkap atau ID video')
    credential=st.text_input('API key YouTube' if platform=='YouTube' else 'Research API access token',type='password')
    limit=st.number_input('Target komentar',min_value=1,max_value=2000,value=100,step=1)
    order=st.selectbox('Urutan YouTube',['time','relevance']) if platform=='YouTube' else 'API'
    permitted=st.checkbox('Saya memiliki izin menggunakan API dan data ini untuk penelitian.')
    if st.button('Ambil komentar',disabled=not (url and credential and permitted)):
        st.session_state.pop('collected',None)
        try:
            with st.spinner('Mengambil komentar melalui API resmi...'):
                collected,status=collect(platform,url,credential,int(limit),order)
            st.session_state['collected']=(platform,url,collected,status)
        except ValueError as e: st.error(str(e))
    saved=st.session_state.get('collected')
    if saved and saved[0]==platform and saved[1]==url:
        raw=saved[2].copy(); source=platform
        st.info(f'{len(raw)} komentar diperoleh. {saved[3]}')
        st.dataframe(raw.head(10))
        st.download_button('Unduh hasil koleksi CSV',raw.map(safe_cell).to_csv(index=False).encode('utf-8-sig'),'koleksi.csv')
if st.button('Analisis sentimen',type='primary'):
    if raw is None or raw.empty: st.warning('Masukkan data dahulu.')
    else:
        d=raw.dropna(subset=['Komentar']).copy(); d.Komentar=d.Komentar.map(redact).str.strip(); d=d[d.Komentar!='']
        if dedup: d=d.drop_duplicates(subset=['Komentar']+(['Kasus'] if 'Kasus' in d else []))
        d=d.reset_index(drop=True)
        if d.empty: st.warning('Tidak ada komentar.')
        else:
            try:
                model=load_model(); predictions=[]; progress=st.progress(0)
                for i in range(0,len(d),16):
                    predictions.extend(model(d.Komentar.iloc[i:i+16].tolist(),truncation=True,batch_size=4))
                    progress.progress(min((i+16)/len(d),1.0))
                mapping={'positive':'Positif','negative':'Negatif','neutral':'Netral'}
                d.insert(0,'ID',[f'C{i+1:05d}' for i in range(len(d))]); d['Kasus']=d['Kasus'] if 'Kasus' in d else case; d['Proyek']=project; d['Sumber']=source
                d['Sentimen_AI']=[mapping.get(p['label'].lower(),p['label']) for p in predictions]
                d['Skor_Model']=[round(p['score'],4) for p in predictions]
                d['Aspek_Saran']=d.Komentar.map(lambda t:tags(t,ASPECTS)); d['Kode_Saran']=d.Komentar.map(lambda t:tags(t,BEHAVIOR))
                for c in ['Coder_1','Coder_2','Sentimen_Restoran','Sentimen_Kreator']: d[c]=PENDING
                d['Emosi_Saran']=d.Komentar.map(lambda t:tags(t,EMOTIONS)); d['Tema_Peneliti']=''; d['Catatan']=''; d['Model']=MODEL
                # Aspek berbasis klausa: prediksi model umum, bukan model ABSA tervalidasi.
                aspect_rows=[]
                for ident,text in zip(d.ID,d.Komentar):
                    for clause in re.split(r'[.!?;]+|\b(?:tetapi|tapi|namun|sedangkan)\b',text,flags=re.I):
                        names=[k for k,v in ASPECTS.items() if re.search(v,clause,re.I)]
                        if names and clause.strip(): aspect_rows.append((ident,clause.strip(),names))
                ap=[]
                for i in range(0,len(aspect_rows),16):
                    ap.extend(model([x[1] for x in aspect_rows[i:i+16]],truncation=True,batch_size=4))
                st.session_state['aspects']=pd.DataFrame([{'ID':ident,'Klausa':clause,'Aspek':a,'Sentimen_Saran':mapping.get(pred['label'].lower(),pred['label']),'Validasi':PENDING} for (ident,clause,names),pred in zip(aspect_rows,ap) for a in names],columns=['ID','Klausa','Aspek','Sentimen_Saran','Validasi'])
                st.session_state['result']=d; st.session_state['version']=st.session_state.get('version',0)+1
            except Exception as e: st.error(f'Analisis gagal: {e}')
if 'result' not in st.session_state: st.stop()
d=st.session_state.result
st.header('Validasi dan pengodean')
st.caption('Aspek dan kode adalah saran kata kunci, bukan ABSA otomatis. Sentimen per objek diisi peneliti. Skor model bukan akurasi.')
editable=['Coder_1','Coder_2','Sentimen_Restoran','Sentimen_Kreator','Tema_Peneliti','Catatan']
config={c:st.column_config.SelectboxColumn(c,options=[PENDING]+LABELS) for c in editable[:4]}
edited=st.data_editor(d,disabled=[c for c in d.columns if c not in editable],column_config=config,hide_index=True,key=f'editor_{st.session_state.version}',use_container_width=True)
edited['Sentimen_Final']=edited.Coder_1.where(edited.Coder_1!=PENDING,edited.Sentimen_AI)
st.header('Dashboard')
a,b,c=st.columns(3); a.metric('Komentar',len(edited)); b.metric('Divalidasi coder 1',int((edited.Coder_1!=PENDING).sum())); c.metric('Belum divalidasi',int((edited.Coder_1==PENDING).sum()))
summary=edited.Sentimen_Final.value_counts().rename_axis('Sentimen').reset_index(name='Jumlah')
st.caption('Distribusi sementara: label manusia jika tersedia, jika tidak prediksi AI.')
st.plotly_chart(px.pie(summary,names='Sentimen',values='Jumlah',hole=.45,color='Sentimen',color_discrete_map={'Positif':'#159447','Negatif':'#d62728','Netral':'#94a3b8','Campuran':'#e59c24'}),use_container_width=True)
st.subheader('Kode perilaku kandidat')
counts=Counter(k for x in edited.Kode_Saran for k in x.split('; ') if k!='Belum teridentifikasi')
if counts: st.bar_chart(pd.Series(dict(counts.most_common())))
stop=set('yang dan di ke dari ini itu saya aku nya untuk dengan tidak ada juga tapi kalau sudah sangat jadi'.split())
words=Counter(w for t in edited.Komentar for w in re.findall(r'\b[^\W\d_]{3,}\b',t.lower()) if w not in stop)
st.subheader('Frekuensi kata'); freq=pd.DataFrame(words.most_common(30),columns=['Kata','Jumlah']); st.dataframe(freq,hide_index=True)
st.subheader('Analisis aspek berbasis klausa (eksperimental)')
st.caption('Aspek dideteksi dengan aturan, polaritas diprediksi pada klausa, bukan ABSA terlatih. Validasi setiap baris.')
aspects=st.session_state.get('aspects',pd.DataFrame())
if not aspects.empty:
    aspects=st.data_editor(aspects,disabled=['ID','Klausa','Aspek','Sentimen_Saran'],column_config={'Validasi':st.column_config.SelectboxColumn('Validasi',options=[PENDING]+LABELS)},key=f"aspect_{st.session_state.version}",hide_index=True)
    aspects['Sentimen_Final']=aspects.Validasi.where(aspects.Validasi!=PENDING,aspects.Sentimen_Saran)
    ac=aspects.groupby(['Aspek','Sentimen_Final']).size().reset_index(name='Jumlah')
    st.plotly_chart(px.bar(ac,x='Aspek',y='Jumlah',color='Sentimen_Final',barmode='group'))
st.subheader('Indikasi emosi (kata kunci)')
em=Counter(k for x in edited.Emosi_Saran for k in x.split('; ') if k!='Belum teridentifikasi')
if em: st.bar_chart(pd.Series(dict(em)))
st.subheader('Word cloud')
if words:
    wc=WordCloud(width=1000,height=450,background_color='white',random_state=42).generate_from_frequencies(words)
    st.image(wc.to_array())
    png=io.BytesIO(); wc.to_image().save(png,format='PNG'); st.download_button('Unduh word cloud PNG',png.getvalue(),'wordcloud.png')
st.subheader('Perbandingan kasus')
cross=edited.groupby(['Kasus','Sentimen_Final']).size().reset_index(name='Jumlah')
st.plotly_chart(px.bar(cross,x='Kasus',y='Jumlah',color='Sentimen_Final',barmode='group'))
st.subheader('Evaluasi model dan kesepakatan coder')
metrics=[]
valid=edited[edited.Coder_1.isin(LABELS[:3]) & edited.Sentimen_AI.isin(LABELS[:3])]
if len(valid):
    acc=(valid.Coder_1==valid.Sentimen_AI).mean(); st.metric('Accuracy terhadap coder 1 (3 kelas)',f'{acc:.1%}'); metrics.append(('Accuracy',acc,len(valid)))
    cm=pd.crosstab(valid.Coder_1,valid.Sentimen_AI).reindex(index=LABELS[:3],columns=LABELS[:3],fill_value=0); st.dataframe(cm)
    rows=[]
    for l in LABELS[:3]:
        tp=cm.loc[l,l]; p=tp/cm[l].sum() if cm[l].sum() else 0; r=tp/cm.loc[l].sum() if cm.loc[l].sum() else 0; f1=2*p*r/(p+r) if p+r else 0
        rows.append({'Kelas':l,'Precision':p,'Recall':r,'F1':f1,'Support':int(cm.loc[l].sum())})
    st.dataframe(pd.DataFrame(rows),hide_index=True)
else: st.info('Isi Coder_1 untuk evaluasi model.')
pairs=edited[edited.Coder_1.isin(LABELS)&edited.Coder_2.isin(LABELS)]
if len(pairs):
    po=(pairs.Coder_1==pairs.Coder_2).mean(); pe=sum((pairs.Coder_1==l).mean()*(pairs.Coder_2==l).mean() for l in LABELS)
    if pe<1:
        k=(po-pe)/(1-pe); st.metric('Cohen’s Kappa',f'{k:.3f}'); metrics.append(('Kappa',k,len(pairs)))
    else: st.info('Kappa tidak terdefinisi karena kedua coder hanya memakai satu kelas yang sama.')
else: st.info('Isi kedua coder secara independen untuk menghitung Kappa.')
st.caption('Evaluasi ini hanya mencakup baris yang telah diberi label sesuai cakupan metrik. Gunakan sampel evaluasi independen untuk klaim ilmiah.')
buf=io.BytesIO()
with pd.ExcelWriter(buf,engine='openpyxl') as writer:
    edited.map(safe_cell).to_excel(writer,sheet_name='Dataset',index=False)
    summary.to_excel(writer,sheet_name='Ringkasan',index=False)
    freq.map(safe_cell).to_excel(writer,sheet_name='Frekuensi',index=False)
    pd.DataFrame(metrics,columns=['Metrik','Nilai','N']).to_excel(writer,sheet_name='Evaluasi',index=False)
    aspects.map(safe_cell).to_excel(writer,sheet_name='Aspek',index=False)
    cross.map(safe_cell).to_excel(writer,sheet_name='Antarkasus',index=False)
    pd.DataFrame(em.items(),columns=['Emosi','Jumlah']).to_excel(writer,sheet_name='Emosi',index=False)
    pd.DataFrame(counts.items(),columns=['Kode','Jumlah']).to_excel(writer,sheet_name='Perilaku',index=False)
    if len(valid): pd.DataFrame(rows).to_excel(writer,sheet_name='Metrik_per_kelas',index=False); cm.to_excel(writer,sheet_name='Confusion_matrix')
st.download_button('Download hasil Excel',buf.getvalue(),'NetnoSense_PRO.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
report = '\n'.join(['NETNOSENSE PRO 3.0 — LAPORAN DESKRIPTIF',f'Proyek: {edited.Proyek.iloc[0]}',f'N: {len(edited)}',f'Divalidasi: {(edited.Coder_1!=PENDING).sum()}', 'Distribusi sementara (campuran label manusia dan AI):',summary.to_string(index=False),'Kode perilaku kandidat: '+str(dict(counts)),'Indikasi emosi: '+str(dict(em)),'Metrik: '+str(metrics),'Batasan: bukan bukti kausal atau kunjungan aktual. Saran aspek dan perilaku perlu validasi.'])
st.download_button('Unduh laporan deskriptif TXT',report.encode('utf-8'),'laporan_netnosense.txt')
st.caption('Unduh sebelum menutup sesi. Proyek tidak disimpan permanen. API hanya mengambil komentar utama. Jangan publikasikan data yang dapat mengidentifikasi pengguna.')
