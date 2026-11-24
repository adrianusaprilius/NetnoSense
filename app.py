import io, re
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

ASPECTS={'Makanan':r'\b(makanan|rasa|enak|hambar|porsi|tekstur|menu)\b','Harga':r'\b(harga|mahal|murah|worth|terjangkau)\b','Pelayanan':r'\b(pelayanan|pelayan|ramah|antri|antre|lambat)\b','Suasana':r'\b(tempat|suasana|bersih|kotor|nyaman)\b','Kreator':r'\b(vlogger|reviewer|influencer|kreator|endorse|review)\b'}
BEHAVIOR={'Curiosity':r'penasaran|membuktikan|bukti sendiri','Avoidance intention':r'(tidak|nggak|gak|ga)\s+(mau|ingin)\s+(datang|coba|beli|makan)|kapok','Visit intention':r'(mau|ingin|pengen|akan|pasti)\s+(datang|coba|mampir|berkunjung|mencoba)','Purchase intention':r'(mau|ingin|pengen)\s+beli','Reviewer skepticism':r'berlebihan|tidak objektif|nggak objektif|bohong|dibayar','FOMO':r'fomo|takut ketinggalan','Reported experience':r'sudah pernah|pernah makan|sudah mencoba'}
def tags(text,patterns):
    return '; '.join(k for k,v in patterns.items() if re.search(v,text,re.I)) or 'Belum teridentifikasi'

def safe_cell(v):
    return "'"+v if isinstance(v,str) and v.lstrip().startswith(('=','+','-','@')) else v

st.title('NetnoSense PRO')
st.caption('Digital Consumer Research Intelligence · v2.0')
st.warning('Gunakan data yang telah dianonimkan. Aplikasi cloud bukan tempat penyimpanan data pribadi. Prediksi AI dan kode berbasis aturan wajib diperiksa peneliti.')
with st.sidebar:
    st.header('Research Settings')
    project=st.text_input('Nama penelitian','Food Vlogger Research')
    source=st.selectbox('Sumber data',['TikTok','YouTube','Instagram','Google Reviews','X','Lainnya'])
    case=st.text_input('Kode kasus / video','V01')
    dedup=st.checkbox('Hapus duplikat dalam impor ini',False)
    st.caption('Model: '+MODEL)
mode=st.radio('Metode input',['Input manual','Upload Excel/CSV'],horizontal=True)
raw=None
if mode=='Input manual':
    s=st.text_area('Satu komentar anonim per baris',height=180)
    raw=pd.DataFrame({'Komentar':s.splitlines()})
else:
    f=st.file_uploader('Dataset anonim',type=['xlsx','csv'])
    if f:
        try:
            data=pd.read_excel(f) if f.name.lower().endswith('.xlsx') else pd.read_csv(f)
            col=st.selectbox('Kolom komentar',data.columns)
            raw=data[[col]].rename(columns={col:'Komentar'})
        except Exception as e: st.error(str(e))
if st.button('Analisis sentimen',type='primary'):
    if raw is None: st.warning('Masukkan data dahulu.')
    else:
        d=raw.dropna().copy(); d.Komentar=d.Komentar.astype(str).str.strip(); d=d[d.Komentar!='']
        if dedup: d=d.drop_duplicates()
        d=d.reset_index(drop=True)
        if d.empty: st.warning('Tidak ada komentar.')
        else:
            try:
                model=load_model(); predictions=[]; progress=st.progress(0)
                for i in range(0,len(d),16):
                    predictions.extend(model(d.Komentar.iloc[i:i+16].tolist(),truncation=True,batch_size=4))
                    progress.progress(min((i+16)/len(d),1.0))
                mapping={'positive':'Positif','negative':'Negatif','neutral':'Netral'}
                d.insert(0,'ID',[f'C{i+1:05d}' for i in range(len(d))]); d['Kasus']=case; d['Proyek']=project; d['Sumber']=source
                d['Sentimen_AI']=[mapping.get(p['label'].lower(),p['label']) for p in predictions]
                d['Skor_Model']=[round(p['score'],4) for p in predictions]
                d['Aspek_Saran']=d.Komentar.map(lambda t:tags(t,ASPECTS)); d['Kode_Saran']=d.Komentar.map(lambda t:tags(t,BEHAVIOR))
                for c in ['Coder_1','Coder_2','Sentimen_Restoran','Sentimen_Kreator']: d[c]=PENDING
                d['Tema_Peneliti']=''; d['Catatan']=''; d['Model']=MODEL
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
st.download_button('Download hasil Excel',buf.getvalue(),'NetnoSense_PRO.xlsx',mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
st.caption('Unduh sebelum menutup sesi. Versi ini tidak mengambil komentar otomatis dari URL dan tidak menyimpan proyek secara permanen.')
