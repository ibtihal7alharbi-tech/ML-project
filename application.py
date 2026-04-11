import warnings; warnings.filterwarnings("ignore")
import streamlit as st, pandas as pd, numpy as np
import plotly.graph_objects as go, plotly.express as px
import os, ssl, io, urllib.request, json, hashlib, datetime
import joblib  # <-- Added for the Smart Model Loader
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import accuracy_score, r2_score
from sklearn.decomposition import PCA

st.set_page_config(page_title="Wisada", page_icon="🌙", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=Outfit:wght@300;400;500;600&display=swap');
*,*::before,*::after{box-sizing:border-box;}
html,body,[class*="css"]{font-family:'Outfit',sans-serif;}
.stApp{background:radial-gradient(ellipse at 20% 0%,#2d1b69 0%,#1a0f3c 30%,#0e0820 60%,#060412 100%);min-height:100vh;color:#e8e0ff;}
.stApp::before{content:'';position:fixed;top:0;left:0;width:100%;height:100%;background-image:radial-gradient(1px 1px at 15% 20%,rgba(255,255,255,.7) 0%,transparent 100%),radial-gradient(1px 1px at 42% 8%,rgba(255,255,255,.5) 0%,transparent 100%),radial-gradient(1px 1px at 67% 15%,rgba(255,255,255,.6) 0%,transparent 100%),radial-gradient(1px 1px at 85% 5%,rgba(255,255,255,.4) 0%,transparent 100%),radial-gradient(1px 1px at 30% 40%,rgba(255,255,255,.3) 0%,transparent 100%),radial-gradient(1px 1px at 55% 35%,rgba(255,255,255,.5) 0%,transparent 100%),radial-gradient(1.5px 1.5px at 8% 55%,rgba(255,255,255,.6) 0%,transparent 100%),radial-gradient(1.5px 1.5px at 92% 28%,rgba(255,255,255,.5) 0%,transparent 100%);pointer-events:none;z-index:0;}
#MainMenu,footer,header,[data-testid="stSidebar"]{display:none!important;}
[data-testid="stAppViewContainer"]{padding:0;}
[data-testid="stMain"] .block-container{padding:2rem 3rem 5rem;max-width:1100px;margin:0 auto;position:relative;z-index:1;}
h1,h2,h3{font-family:'Cormorant Garamond',serif;}

/* ── base card ── */
.card{padding:1.5rem 1.8rem;border-radius:22px;background:rgba(255,255,255,.03);border:1px solid rgba(192,132,252,.13);margin-bottom:1rem;}
.card-gold{padding:1.5rem 1.8rem;border-radius:22px;background:linear-gradient(135deg,rgba(244,201,122,.09),rgba(192,132,252,.06));border:1px solid rgba(244,201,122,.22);margin-bottom:1rem;}
.card-purple{padding:1.5rem 1.8rem;border-radius:22px;background:rgba(124,58,237,.08);border:1px solid rgba(192,132,252,.2);margin-bottom:1rem;}
.card-red{padding:1.5rem 1.8rem;border-radius:22px;background:rgba(248,113,113,.05);border:1px solid rgba(248,113,113,.2);margin-bottom:1rem;}

/* ── metric block ── */
.metric-block{padding:1.8rem 1.4rem;border-radius:22px;background:rgba(255,255,255,.03);border:1px solid rgba(192,132,252,.13);height:100%;}
.mb-label{font-size:.62rem;letter-spacing:2.5px;color:#6d5fa0;text-transform:uppercase;font-weight:600;margin-bottom:.5rem;}
.mb-num{font-family:'Cormorant Garamond',serif;font-size:3.2rem;font-weight:300;line-height:1;margin-bottom:.3rem;}
.mb-word{font-family:'Cormorant Garamond',serif;font-size:1rem;font-style:italic;margin-bottom:.6rem;}
.mb-bar-wrap{background:rgba(255,255,255,.06);border-radius:99px;height:5px;overflow:hidden;margin:.5rem 0;}
.mb-bar{height:100%;border-radius:99px;transition:width .3s;}
.mb-desc{font-size:.78rem;color:#7c6fa0;line-height:1.7;margin-top:.6rem;border-top:1px solid rgba(192,132,252,.1);padding-top:.6rem;}

/* ── bedtime ── */
.bedtime-hero{text-align:center;background:linear-gradient(160deg,rgba(244,201,122,.12),rgba(192,132,252,.07));border:1px solid rgba(244,201,122,.28);border-radius:24px;padding:2.2rem 1.5rem;}
.bt-clock{font-family:'Cormorant Garamond',serif;font-size:4.5rem;color:#f4c97a;font-weight:300;line-height:1;letter-spacing:2px;}
.bt-tag{font-size:.65rem;letter-spacing:2.5px;color:#9d8ec7;text-transform:uppercase;margin-bottom:.5rem;}
.bt-detail{font-size:.83rem;color:#9d8ec7;line-height:1.75;margin-top:.8rem;}

/* ── tips ── */
.tip-row{display:flex;gap:1.1rem;align-items:flex-start;padding:1.3rem 1.5rem;border-radius:18px;background:rgba(192,132,252,.05);border:1px solid rgba(192,132,252,.13);margin-bottom:.8rem;}
.tip-icon{font-size:1.4rem;flex-shrink:0;padding-top:1px;}
.tip-title{font-size:.92rem;color:#e8e0ff;font-weight:600;margin-bottom:.3rem;}
.tip-body{font-size:.82rem;color:#9d8ec7;line-height:1.7;}
.tip-action{display:inline-block;margin-top:.5rem;font-size:.72rem;letter-spacing:1px;color:#c084fc;border:1px solid rgba(192,132,252,.3);border-radius:99px;padding:2px 10px;}

/* ── article (explore) ── */
.article-section{margin-bottom:2.5rem;}
.article-heading{font-family:'Cormorant Garamond',serif;font-size:1.6rem;color:#e8e0ff;font-weight:400;margin-bottom:.6rem;}
.article-body{font-size:.88rem;color:#8a7db8;line-height:1.9;}
.article-stat{display:inline-block;background:rgba(192,132,252,.1);border:1px solid rgba(192,132,252,.2);border-radius:10px;padding:.4rem 1rem;font-size:.82rem;color:#c084fc;margin:.3rem .3rem .3rem 0;}

/* ── weekly history ── */
.week-day{padding:1rem;border-radius:16px;background:rgba(255,255,255,.03);border:1px solid rgba(192,132,252,.1);text-align:center;}
.week-score{font-family:'Cormorant Garamond',serif;font-size:2rem;font-weight:300;}
.week-date{font-size:.68rem;color:#6d5fa0;letter-spacing:1px;margin-bottom:.4rem;}

/* ── auth ── */
.auth-wrap{max-width:460px;margin:1.5rem auto 0;background:rgba(45,27,105,.2);border:1px solid rgba(192,132,252,.2);border-radius:28px;padding:2.5rem;}
.form-section{font-family:'Cormorant Garamond',serif;font-size:1.2rem;color:#c084fc;margin:.8rem 0;}
.hint-box{background:rgba(192,132,252,.07);border:1px solid rgba(192,132,252,.18);border-radius:12px;padding:.8rem 1rem;margin-bottom:.9rem;font-size:.8rem;color:#9d8ec7;line-height:1.7;}
.divider{height:1px;background:linear-gradient(90deg,transparent,rgba(192,132,252,.22),transparent);margin:1.8rem 0;}

/* ── widgets ── */
.stSlider>label,.stSelectbox>label,.stNumberInput>label,.stTextInput>label{color:#9d8ec7!important;font-size:.77rem!important;letter-spacing:.8px!important;}
.stSlider [data-baseweb="thumb"]{background:#c084fc!important;}
.stSlider [data-baseweb="track-fill"]{background:linear-gradient(90deg,#7c3aed,#c084fc)!important;}
.stSlider [data-baseweb="track"]{background:rgba(192,132,252,.15)!important;}
.stSelectbox [data-baseweb="select"]>div,.stNumberInput input,.stTextInput input{background:rgba(255,255,255,.04)!important;border:1px solid rgba(192,132,252,.2)!important;border-radius:12px!important;color:#e8e0ff!important;}
.stButton>button{background:linear-gradient(135deg,#7c3aed,#9333ea)!important;color:#fff!important;border:none!important;border-radius:14px!important;font-family:'Outfit',sans-serif!important;font-size:.87rem!important;font-weight:500!important;padding:.75rem 2rem!important;letter-spacing:.5px!important;box-shadow:0 4px 24px rgba(124,58,237,.35)!important;}
.stButton>button:hover{transform:translateY(-1px)!important;box-shadow:0 8px 32px rgba(124,58,237,.5)!important;}
.stTabs [data-baseweb="tab-list"]{background:rgba(255,255,255,.03)!important;border-radius:14px!important;padding:4px!important;gap:4px!important;border:1px solid rgba(192,132,252,.12)!important;}
.stTabs [data-baseweb="tab"]{color:#7c6fa0!important;border-radius:10px!important;font-family:'Outfit',sans-serif!important;font-size:.82rem!important;}
.stTabs [aria-selected="true"]{background:rgba(124,58,237,.35)!important;color:#e8e0ff!important;}
hr{border:none;border-top:1px solid rgba(192,132,252,.12);margin:2rem 0;}
[data-testid="stFormSubmitButton"]>button{width:100%!important;padding:1rem 2rem!important;font-size:1rem!important;}
</style>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════
#  USER STORE
# ═══════════════════════════════════════════════════════════
UF = Path(os.path.dirname(os.path.abspath(__file__))) / "wisada_users.json"
def _lu(): return json.load(open(UF)) if UF.exists() else {}
def _su(u): json.dump(u, open(UF,"w"), indent=2)
def _h(pw): return hashlib.sha256(pw.encode()).hexdigest()

def register(name,email,pw,profile):
    u=_lu()
    if email in u: return False,"Account already exists."
    u[email]={"name":name,"password":_h(pw),"profile":profile,"history":[]}
    _su(u); return True,"ok"

def do_login(email,pw):
    u=_lu()
    if email not in u: return False,None,None,None,"No account found."
    if u[email]["password"]!=_h(pw): return False,None,None,None,"Wrong password."
    return True,u[email]["name"],u[email].get("profile",{}),u[email].get("history",[]),"ok"

def save_entry(email,entry):
    u=_lu()
    if email in u:
        u[email].setdefault("history",[]).append(entry); _su(u)

# ═══════════════════════════════════════════════════════════
#  DATA
# ═══════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def load_data():
    sd=os.path.dirname(os.path.abspath(__file__))
    for p in [os.path.join(sd,"sleep_health_dataset.csv"),os.path.join(sd,"..","sleep_health_dataset.csv")]:
        if os.path.exists(p):
            df=pd.read_csv(p); df.columns=df.columns.str.strip()
            for c in df.select_dtypes("object"): df[c]=df[c].str.lower().str.strip()
            df.drop(columns=["person_id"],inplace=True,errors="ignore"); return df
    url="https://raw.githubusercontent.com/ibtihal7alharbi-tech/ML-project/refs/heads/main/sleep_health_dataset.csv"
    ctx=ssl.create_default_context(); ctx.check_hostname=False; ctx.verify_mode=ssl.CERT_NONE
    with urllib.request.urlopen(url,context=ctx) as r: df=pd.read_csv(io.BytesIO(r.read()))
    df.columns=df.columns.str.strip()
    for c in df.select_dtypes("object"): df[c]=df[c].str.lower().str.strip()
    df.drop(columns=["person_id"],inplace=True,errors="ignore"); return df

# ═══════════════════════════════════════════════════════════
#  MODELS & SMART LOADER
# ═══════════════════════════════════════════════════════════
UNUM=["age","bmi","sleep_duration_hrs","sleep_latency_mins","wake_episodes_per_night",
      "rem_percentage","deep_sleep_percentage","nap_duration_mins","stress_score",
      "exercise_day","work_hours_that_day","caffeine_mg_before_bed",
      "screen_time_before_bed_mins","room_temperature_celsius","steps_that_day",
      "weekend_sleep_diff_hrs","sleep_aid_used","shift_work",
      "restorative_sleep_score","sleep_disruption_score","pre_bed_stimulant_load","sleep_debt_proxy"]
UCAT=["gender","occupation","chronotype","mental_health_condition"]

def _train_all_from_scratch(df_raw):
    df=df_raw.copy(); M={}
    # feature engineering
    df["restorative_sleep_score"]=df["rem_percentage"]+df["deep_sleep_percentage"]
    df["pre_bed_stimulant_load"]=(df["caffeine_mg_before_bed"]/(df["caffeine_mg_before_bed"].max()+1)
                                  +df["alcohol_units_before_bed"]/(df["alcohol_units_before_bed"].max()+1))
    df["sleep_debt_proxy"]=df["weekend_sleep_diff_hrs"].abs()
    df["sleep_disruption_score"]=(df["stress_score"]/(df["stress_score"].max()+1)
                                  +df["wake_episodes_per_night"]/(df["wake_episodes_per_night"].max()+1)
                                  +df["sleep_latency_mins"]/(df["sleep_latency_mins"].max()+1))
    M["maxes"]={c:float(df[c].max()) for c in ["caffeine_mg_before_bed","alcohol_units_before_bed",
                "stress_score","wake_episodes_per_night","sleep_latency_mins"]}

    # 1. Sleep Disorder
    df_d=df.copy(); le_d={}
    for c in df_d.select_dtypes("object").columns:
        if c=="sleep_disorder_risk": continue
        le=LabelEncoder(); df_d[c]=le.fit_transform(df_d[c].astype(str)); le_d[c]=le
    tgt=LabelEncoder(); df_d["sleep_disorder_risk"]=tgt.fit_transform(df_d["sleep_disorder_risk"])
    drop_d=[c for c in ["sleep_quality_score","cognitive_performance_score","felt_rested"] if c in df_d.columns]
    X_d=df_d.drop(columns=["sleep_disorder_risk"]+drop_d).select_dtypes("number"); y_d=df_d["sleep_disorder_risk"]
    Xt,Xe,yt,ye=train_test_split(X_d,y_d,test_size=.2,random_state=42,stratify=y_d)
    m=RandomForestClassifier(n_estimators=100,max_depth=10,class_weight="balanced",random_state=42); m.fit(Xt,yt)
    M.update(dis_m=m,dis_le=le_d,dis_tgt=tgt,dis_cols=list(X_d.columns),dis_acc=accuracy_score(ye,m.predict(Xe)))

    # 2. Sleep Quality
    drop_sq=[c for c in ["sleep_quality_score","cognitive_performance_score","sleep_disorder_risk","felt_rested"] if c in df.columns]
    dq=df.drop(columns=drop_sq,errors="ignore").copy(); dq["__y"]=df["sleep_quality_score"]; dq.dropna(inplace=True)
    cat_q=dq.drop(columns=["__y"]).select_dtypes("object").columns.tolist()
    Xq=pd.get_dummies(dq.drop(columns=["__y"]),columns=cat_q,drop_first=False); yq=dq["__y"]
    Xt,Xe,yt,ye=train_test_split(Xq,yq,test_size=.15,random_state=42)
    Xt2,_,yt2,_=train_test_split(Xt,yt,test_size=.1765,random_state=42)
    mq=RandomForestRegressor(n_estimators=100,random_state=42,n_jobs=-1); mq.fit(Xt2,yt2)
    M.update(sq_m=mq,sq_cols=list(Xq.columns),sq_cat=cat_q,sq_r2=r2_score(ye,mq.predict(Xe)))

    # 3. Felt Rested
    drop_fr=[c for c in ["felt_rested","sleep_disorder_risk","cognitive_performance_score"] if c in df.columns]
    dr=df.drop(columns=drop_fr,errors="ignore").copy(); dr["__y"]=df["felt_rested"]; dr.dropna(inplace=True)
    cat_r=dr.drop(columns=["__y"]).select_dtypes("object").columns.tolist()
    Xr=pd.get_dummies(dr.drop(columns=["__y"]),columns=cat_r,drop_first=False); yr=dr["__y"]
    Xt,Xe,yt,ye=train_test_split(Xr,yr,test_size=.2,random_state=42,stratify=yr)
    mr=RandomForestClassifier(n_estimators=100,max_depth=10,class_weight="balanced",random_state=42); mr.fit(Xt,yt)
    M.update(fr_m=mr,fr_cols=list(Xr.columns),fr_cat=cat_r,fr_acc=accuracy_score(ye,mr.predict(Xe)))

    # 4. Cognitive
    cnum=[c for c in ["age","bmi","sleep_duration_hrs","rem_percentage","deep_sleep_percentage",
                       "sleep_latency_mins","wake_episodes_per_night","exercise_day","nap_duration_mins",
                       "stress_score","work_hours_that_day","caffeine_mg_before_bed"] if c in df.columns]
    ccat=[c for c in ["gender","chronotype","mental_health_condition","sleep_aid_used","shift_work"] if c in df.columns]
    dc=df[cnum+ccat+["cognitive_performance_score"]].dropna().copy()
    if "gender" in dc.columns: dc=dc[~dc["gender"].isin(["other",""])]
    Xc=pd.get_dummies(dc.drop(columns=["cognitive_performance_score"]),columns=ccat,drop_first=False); yc=dc["cognitive_performance_score"]
    Xt,Xe,yt,ye=train_test_split(Xc,yc,test_size=.3,random_state=42)
    mc=HistGradientBoostingRegressor(max_iter=200,learning_rate=.05,max_depth=6,min_samples_leaf=20,random_state=42); mc.fit(Xt,yt)
    M.update(cog_m=mc,cog_cols=list(Xc.columns),cog_cat=ccat,cog_r2=r2_score(ye,mc.predict(Xe)))

    # 5. PCA
    pf=[c for c in ["stress_score","sleep_duration_hrs","caffeine_mg_before_bed","screen_time_before_bed_mins",
                     "exercise_day","work_hours_that_day","room_temperature_celsius","bmi",
                     "wake_episodes_per_night","sleep_latency_mins","rem_percentage",
                     "deep_sleep_percentage","nap_duration_mins","steps_that_day"] if c in df.columns]
    dp=df[pf+["sleep_quality_score"]].dropna(); Xp=dp[pf]
    sc=StandardScaler(); Xs=sc.fit_transform(Xp)
    pca=PCA(n_components=6,random_state=42); comps=pca.fit_transform(Xs)
    pcn=[f"PC{i+1}" for i in range(6)]
    load=pd.DataFrame(pca.components_.T,index=pf,columns=pcn)
    pdfr=pd.DataFrame(comps,columns=pcn); pdfr["sq"]=dp["sleep_quality_score"].values
    corrs=pdfr.drop(columns=["sq"]).corrwith(pdfr["sq"],method="spearman")
    M.update(pca=pca,pca_sc=sc,pca_load=load,pca_corrs=corrs,pca_feats=pf,pca_def=Xp.mean().to_dict(),pc_names=pcn)
    return M

@st.cache_resource(show_spinner="Loading AI Models (Only takes a few seconds the first time)...")
def get_cached_models(_df_raw):
    model_file = "wisada_pretrained_models.joblib"
    if os.path.exists(model_file):
        return joblib.load(model_file)
    else:
        M = _train_all_from_scratch(_df_raw)
        joblib.dump(M, model_file)
        return M

# ═══════════════════════════════════════════════════════════
#  PREDICT
# ═══════════════════════════════════════════════════════════
def predict(u,M,df):
    out={}
    # disorder
    row={}
    for c in M["dis_cols"]:
        if c in M["dis_le"]:
            v=str(u.get(c,"")).lower().strip(); le=M["dis_le"][c]
            row[c]=int(le.transform([v])[0]) if v in le.classes_ else 0
        else:
            try: row[c]=float(u.get(c,0))
            except: row[c]=0.0
    Xd=pd.DataFrame([row])[M["dis_cols"]].fillna(0).astype(float)
    enc=M["dis_m"].predict(Xd)[0]; prb=M["dis_m"].predict_proba(Xd)[0]
    out["disorder"]=M["dis_tgt"].inverse_transform([enc])[0]
    out["disorder_proba"]=dict(zip(M["dis_tgt"].classes_,prb))
    out["none_prob"]=float(out["disorder_proba"].get("none",.5))

    # sleep quality
    sq_drop=[c for c in ["sleep_quality_score","cognitive_performance_score","sleep_disorder_risk","felt_rested"] if c in df.columns]
    row_sq={c:u.get(c,0) for c in df.columns if c not in sq_drop}
    ts=pd.get_dummies(pd.DataFrame([row_sq]),columns=M["sq_cat"],drop_first=False)
    ts=ts.reindex(columns=M["sq_cols"],fill_value=0)
    for c in ts.columns: ts[c]=pd.to_numeric(ts[c],errors="coerce").fillna(0)
    out["sleep_quality"]=float(np.clip(M["sq_m"].predict(ts)[0],1,10))

    # felt rested
    fr_drop=[c for c in ["felt_rested","sleep_disorder_risk","cognitive_performance_score"] if c in df.columns]
    row_fr={c:u.get(c,0) for c in df.columns if c not in fr_drop}
    tf=pd.get_dummies(pd.DataFrame([row_fr]),columns=M["fr_cat"],drop_first=False)
    tf=tf.reindex(columns=M["fr_cols"],fill_value=0)
    for c in tf.columns: tf[c]=pd.to_numeric(tf[c],errors="coerce").fillna(0)
    out["felt_rested"]=bool(M["fr_m"].predict(tf)[0])

    # cognitive
    cnum=[c for c in ["age","bmi","sleep_duration_hrs","rem_percentage","deep_sleep_percentage",
                       "sleep_latency_mins","wake_episodes_per_night","exercise_day","nap_duration_mins",
                       "stress_score","work_hours_that_day","caffeine_mg_before_bed"] if c in df.columns]
    row_c={c:float(u.get(c,0)) for c in cnum}
    for c in M["cog_cat"]: row_c[c]=str(u.get(c,"")).lower().strip()
    tc=pd.get_dummies(pd.DataFrame([row_c]),columns=M["cog_cat"],drop_first=False)
    tc=tc.reindex(columns=M["cog_cols"],fill_value=0)
    for c in tc.columns: tc[c]=pd.to_numeric(tc[c],errors="coerce").fillna(0)
    out["cognitive"]=float(np.clip(M["cog_m"].predict(tc)[0],30,100))

    # bedtime
    sq=out["sleep_quality"]; ideal=7.0 if sq>=7 else 7.5 if sq>=5 else 8.5
    wh=int(u.get("wake_hour",7)); wm=int(u.get("wake_minute",0))
    total=(wh*60+wm)-int(ideal*60)
    if total<0: total+=1440
    out["bedtime"]=f"{(total//60)%24:02d}:{total%60:02d}"
    out["ideal_hrs"]=ideal; out["wake_str"]=f"{wh:02d}:{wm:02d}"

    # pca saboteurs
    pr={f:float(u.get(f,M["pca_def"].get(f,0))) for f in M["pca_feats"]}
    ps=M["pca_sc"].transform(pd.DataFrame([pr])[M["pca_feats"]].fillna(0))
    pc=M["pca"].transform(ps)[0]; sab=[]
    for i,pn in enumerate(M["pc_names"]):
        c2=M["pca_corrs"][pn]
        if (c2<0 and pc[i]>0) or (c2>0 and pc[i]<0):
            top=M["pca_load"][pn].abs().sort_values(ascending=False).head(2).index.tolist()
            sab.append({"features":top,"impact":round(abs(float(c2))*100,1)})
    sab.sort(key=lambda x:x["impact"],reverse=True); out["saboteurs"]=sab[:2]
    out["score"]=round(np.clip(out["sleep_quality"]*4+out["cognitive"]*.25+(18 if out["felt_rested"] else 0)+out["none_prob"]*8,0,100))
    return out

# ═══════════════════════════════════════════════════════════
#  CONTENT GENERATORS
# ═══════════════════════════════════════════════════════════
def sq_label(sq):
    if sq>=9: return "Perfect","#a78bfa"
    if sq>=8: return "Excellent","#a78bfa"
    if sq>=7: return "Good","#7dd3a8"
    if sq>=6: return "Decent","#f4c97a"
    if sq>=5: return "Moderate","#f4c97a"
    if sq>=3.5: return "Poor","#f87171"
    return "Very Poor","#f87171"

def cog_label(cog):
    if cog>=85: return "Peak Focus","#a78bfa"
    if cog>=75: return "Sharp","#a78bfa"
    if cog>=65: return "Good","#7dd3a8"
    if cog>=55: return "Reduced","#f4c97a"
    if cog>=45: return "Foggy","#f87171"
    return "Very Low","#f87171"

def score_label(s):
    if s>=82: return "Excellent","#a78bfa"
    if s>=68: return "Good","#7dd3a8"
    if s>=52: return "Fair","#f4c97a"
    return "Needs Care","#f87171"

def sq_full_desc(sq,hrs,lat,wk,deep,stress,caf,screen,exercise,temp):
    lines=[]
    if sq>=8: lines.append(f"Your sleep was deeply restorative. A score of {sq:.1f}/10 puts you in the top tier — your body completed full sleep cycles and your brain consolidated memories effectively.")
    elif sq>=7: lines.append(f"Solid night at {sq:.1f}/10. You hit the threshold for restorative sleep, meaning your body went through sufficient deep and REM cycles.")
    elif sq>=5.5: lines.append(f"Moderate quality at {sq:.1f}/10. Your body recovered partially but some sleep stages were likely cut short or interrupted.")
    elif sq>=4: lines.append(f"Below average at {sq:.1f}/10. Your body didn't get the full recovery it needed. You may feel the effects in your energy and focus today.")
    else: lines.append(f"Very poor at {sq:.1f}/10. Multiple factors disrupted your sleep architecture last night. This is worth addressing tonight.")

    if hrs>=8.5 and sq<7: lines.append(f"Interesting — you slept {hrs:.1f} hours but quality was still low. Long sleep with poor quality often points to fragmented cycles or insufficient deep sleep, not just hours in bed.")
    elif hrs<6: lines.append(f"Only {hrs:.1f} hours cuts your sleep cycles short. Each full cycle takes ~90 min, so you need at least 5 complete cycles (7.5 hrs) for full restoration.")
    elif hrs>=9: lines.append(f"You slept {hrs:.1f} hours. While rest is important, consistently sleeping over 9 hours can indicate accumulated sleep debt or other factors worth monitoring.")

    contributors=[]
    if lat>25: contributors.append(f"long sleep onset ({lat:.0f} min)")
    if wk>=2: contributors.append(f"frequent wakings ({wk}x)")
    if deep<15: contributors.append("insufficient deep sleep")
    if stress>=7: contributors.append(f"high pre-bed stress ({stress}/10)")
    if caf>100: contributors.append(f"caffeine ({caf}mg)")
    if screen>45: contributors.append(f"screen exposure ({screen} min)")
    if temp>22: contributors.append(f"warm bedroom ({temp}°C)")
    if contributors: lines.append("Key contributors to your score: " + ", ".join(contributors) + ".")
    return " ".join(lines)

def cog_full_desc(cog,sq,hrs,stress):
    if cog>=80: return f"At {cog:.0f}/100, your cognitive performance is genuinely high today. Your prefrontal cortex — responsible for decisions, creativity, and emotional control — had sufficient restoration time. Expect strong focus, faster problem-solving, and better emotional regulation."
    if cog>=65: return f"At {cog:.0f}/100, you're in decent shape mentally. You'll be functional and productive, though complex tasks or sustained deep work may feel slightly harder than usual, especially after 3 PM when your alertness naturally dips."
    if cog>=50: return f"At {cog:.0f}/100, your cognitive resources are limited today. Sleep deprivation reduces working memory capacity and increases error rates. Prioritise important decisions for your peak hours and take short breaks every 45 minutes."
    return f"At {cog:.0f}/100, expect a genuinely difficult day mentally. Studies show that below 5 hours of quality sleep, reaction time is comparable to being mildly intoxicated. Avoid major decisions, and if possible, a 20-minute nap between 1–3 PM can partially restore function."

def rested_full_desc(rested,sq,hrs,wk):
    if rested: return f"The model predicts you'll feel rested when you wake. This is consistent with your sleep quality ({sq:.1f}/10) and duration ({hrs:.1f} hrs). Your sleep architecture appears sufficient for morning alertness."
    if hrs<5.5: return f"With {hrs:.1f} hours of sleep, your body hasn't completed enough cycles to feel restored. Sleep inertia — that groggy, disoriented feeling — will likely persist for 20–40 minutes after waking."
    if wk>=3: return f"Waking up {wk} times prevented your sleep from deepening properly. Even if the total hours look reasonable, fragmented sleep leaves you feeling just as tired as a short night."
    if sq<5: return f"Your sleep quality score ({sq:.1f}/10) indicates your sleep stages were disrupted. Even with enough hours in bed, poor-quality sleep leaves your adenosine clearance incomplete — the chemical that makes you feel tired will still be elevated."
    return "You may feel slightly groggy initially. Your body was in a lighter sleep stage at wake time. Give it 15–20 minutes and you should feel more alert."

def dis_full_desc(disorder,none_prob,stress,wk,lat):
    d=disorder.lower()
    if d=="none":
        conf=int(none_prob*100)
        if conf>=80: return f"No signs of a sleep disorder detected ({conf}% confidence). Your sleep pattern is consistent with healthy sleep architecture."
        return f"No disorder detected ({conf}% confidence). Your pattern is generally healthy, though some habits (like high stress or frequent wakings) are worth monitoring over time."
    if "insomnia" in d:
        reasons=[]
        if lat>25: reasons.append(f"long sleep onset ({lat:.0f} min suggests difficulty initiating sleep)")
        if wk>=3: reasons.append(f"frequent night wakings ({wk}x)")
        if stress>=7: reasons.append("high stress level")
        base="Your pattern shows markers consistent with insomnia risk."
        if reasons: base+=" Contributing signals: "+", ".join(reasons)+"."
        base+=" Insomnia is highly treatable — Cognitive Behavioral Therapy for Insomnia (CBT-I) has an 80% success rate, higher than sleep medications."
        return base
    if "apnea" in d:
        return f"Your profile shows markers associated with sleep apnea risk — particularly frequent wakings ({wk}x) and non-restorative sleep despite adequate hours. Sleep apnea causes repeated micro-arousals that prevent deep sleep. A sleep study (polysomnography) can confirm this definitively."
    return "Your sleep pattern shows some irregularities worth addressing. Consistent sleep hygiene improvements over 2–3 weeks often resolve many sleep quality issues."

def bedtime_full_explanation(p,u):
    sq=p["sleep_quality"]; ideal=p["ideal_hrs"]
    wh=u.get("wake_hour",7); wm=u.get("wake_minute",0)
    wake_fmt=f"{wh:02d}:{wm:02d}"
    lines=[f"Based on your sleep quality score of {sq:.1f}/10, your body needs approximately {ideal:.1f} hours tonight to complete its recovery cycles."]
    if sq>=7: lines.append("With good sleep quality, your sleep cycles are efficient — 7 hours gives you roughly 5 complete 90-minute cycles, which is the sweet spot for most adults.")
    elif sq>=5: lines.append("With moderate quality sleep, aim for 7.5 hours to compensate for the efficiency loss — incomplete or disrupted cycles mean your body needs a bit more time in bed.")
    else: lines.append("After a difficult night, your body needs 8.5 hours to attempt to rebuild. Poor quality sleep cycles are shorter and less restorative, so more time in bed is necessary.")
    lines.append(f"Working backwards from your {wake_fmt} wake time gives you a target bedtime of {p['bedtime']}.")
    lines.append("Set a reminder 30 minutes before this time to begin your wind-down. Consistency in your sleep-wake times is the single most powerful way to improve sleep quality over time — it anchors your circadian rhythm.")
    return " ".join(lines)

# Each entry: (display_name, [bullet1, bullet2, bullet3], action_text)
SAB_MAP={
    "stress_score":("You were stressed today",
        ["Stress keeps your brain in alert mode — the opposite of what you need to fall into deep sleep",
         "When you're stressed, your body produces a hormone that blocks the sleep hormone (melatonin)",
         "The more stressed you were, the lighter and more restless your sleep will be tonight"],
        "Before bed: write down everything on your mind on paper, close the notebook. Then try slow breathing — 4 seconds in, hold 4, out 6. Repeat 5 times."),
    "sleep_duration_hrs":("You didn't sleep enough",
        ["Every hour under 7 is one less recovery cycle — your body didn't finish repairing itself",
         "Sleep debt builds up over days — last night's short sleep makes tonight's even more important",
         "Your memory, mood, and focus are all directly affected by total sleep time"],
        "Set a bedtime alarm tonight — not just a morning alarm. Commit to it like a scheduled appointment."),
    "caffeine_mg_before_bed":("Caffeine is still in your body",
        ["Caffeine blocks the signal that tells your brain it's time to sleep — it stays active for 6+ hours",
         "Even if you fall asleep, caffeine reduces the depth of your sleep without you realising",
         "The more caffeine you had, the lighter your sleep will likely be tonight"],
        "Tomorrow's rule: nothing caffeinated after 1 PM. Switch to water, herbal tea, or decaf after that."),
    "screen_time_before_bed_mins":("Too much screen time before bed",
        ["The light from screens tells your brain it's still daytime — delaying your natural tiredness",
         "Scrolling keeps your mind active and stimulated right when it should be slowing down",
         "The more screen time before bed, the harder it is to fall into deep sleep"],
        "Put your phone in another room 45 min before bed. Use a standalone alarm clock — you don't need your phone in the bedroom."),
    "wake_episodes_per_night":("You kept waking up at night",
        ["Each time you wake up, your sleep cycle resets back to the beginning",
         "The best deep sleep happens in the middle of the night — repeated wakings cut it short",
         "Even short wakings you don't remember reduce how rested you feel in the morning"],
        "Check one thing tonight: is your room too warm? Too much light? A noise? Fluids before bed? Fix the one that most likely applies."),
    "sleep_latency_mins":("It took too long to fall asleep",
        ["Taking over 30 minutes to fall asleep means your mind was still too active at bedtime",
         "The longer this happens regularly, your brain starts associating bed with being awake",
         "This is fixable — it usually comes from stress, screens, or no wind-down routine"],
        "If you're still awake after 20 minutes, get up. Do something calm and boring in dim light. Go back to bed only when you actually feel sleepy."),
    "rem_percentage":("Your deep/REM sleep was low",
        ["REM sleep is when your brain processes emotions, stores memories, and recovers mentally",
         "Low REM leaves you feeling emotionally drained and forgetful the next day",
         "Stress, caffeine, and interruptions all reduce your REM sleep time"],
        "Protect your last 2 hours of sleep — don't set your alarm earlier than you need to. That's when most REM happens."),
    "deep_sleep_percentage":("Your deep sleep was low",
        ["Deep sleep is when your body physically heals itself — muscles, immune system, energy",
         "Without enough deep sleep, you wake up feeling tired even after a full night",
         "Most deep sleep happens in the first half of the night — going to bed earlier helps"],
        "Exercise earlier in the day — not right before bed. It's the most reliable way to get more deep sleep."),
    "exercise_day":("You didn't exercise today",
        ["Exercise makes your body physically tired — which is what drives deep, quality sleep",
         "Without any movement today, your body has less reason to sleep deeply tonight",
         f"Low movement + low steps = lighter sleep tonight"],
        "Tomorrow: a 25-minute walk before 6 PM. It's the simplest thing you can do to improve tonight's sleep score."),
    "room_temperature_celsius":("Your bedroom was too warm",
        ["Your body needs to cool down slightly to enter deep sleep — a warm room prevents this",
         "Sleeping in a warm room keeps you in lighter sleep stages all night",
         "This is one of the easiest and most effective fixes for better sleep"],
        "Open a window, use a fan, or set AC to 18–20°C before you sleep. Keep your feet outside the covers to cool down faster."),
    "work_hours_that_day":("Long work day",
        ["After long hours of work, your brain stays in 'work mode' for hours — making it hard to switch off",
         "This makes it harder to fall asleep and results in lighter early-night sleep",
         "Your brain needs a clear break between work and sleep"],
        "After work: close everything, write tomorrow's 3 priorities, then stop. No emails, no news, no work after that point."),
    "steps_that_day":("Low movement today",
        ["Not moving much means your body isn't physically tired at bedtime",
         "Your body uses physical tiredness as one of the signals to sleep deeply",
         "Desk-bound days almost always result in lighter, less satisfying sleep"],
        "Tomorrow: take the stairs, walk during calls, park further. Reaching 7,000 steps makes a real difference."),
    "nap_duration_mins":("Your nap was too long",
        ["Naps over 30 minutes enter deep sleep — waking from deep sleep mid-day leaves you groggy",
         "A long nap also reduces how tired you feel at bedtime, making it harder to sleep well at night",
         "You're essentially spending some of tonight's sleep budget during the day"],
        "Keep naps to 20 minutes max, before 3 PM. Set an alarm before lying down."),
}

def get_personalized_tips(u,p):
    tips=[]
    hrs=u.get("sleep_duration_hrs",7); sq=p["sleep_quality"]
    stress=u.get("stress_score",5); caf=u.get("caffeine_mg_before_bed",0)
    screen=u.get("screen_time_before_bed_mins",0); ex=u.get("exercise_day",1)
    wk=u.get("wake_episodes_per_night",1); lat=u.get("sleep_latency_mins",15)
    temp=u.get("room_temperature_celsius",20); nap=u.get("nap_duration_mins",0)
    steps=u.get("steps_that_day",7000); mh=str(u.get("mental_health_condition","none")).lower()
    work=u.get("work_hours_that_day",8)

    if hrs<5.5:
        tips.append(("◑",f"You only slept {hrs:.1f} hours — your body needs more",
            f"That's {7-hrs:.1f} hours less than the minimum your body needs to fully recover. Today you'll feel it in your mood, memory, and energy.",
            f"Non-negotiable tonight: be in bed by {p['bedtime']}. Set the alarm now."))
    elif hrs<7:
        tips.append(("◑",f"You're {(7-hrs):.1f} hour(s) short of a full recovery",
            f"At {hrs:.1f} hours, your body didn't finish its full repair cycle. Over a week of this, the effect adds up to losing a whole night's sleep.",
            "Move your bedtime 30 minutes earlier tonight. Keep doing it until you hit 7+ hours."))

    if stress>=8:
        tips.append(("✦",f"Your stress today ({stress}/10) will affect your sleep",
            f"When you're this stressed, your body stays in 'alert mode' even when you lie down. You'll likely wake up more and get less deep sleep.",
            "Before bed: write down everything on your mind — every task, worry, thought. Close the notebook. This moves the mental load out of your head so your brain can rest."))
    elif stress>=6:
        tips.append(("✦",f"You had a stressful day ({stress}/10) — wind down properly",
            "Stress makes your brain stay active even when your eyes are closed. Without a real wind-down, you'll spend more time in light sleep.",
            "30 min before bed: no work, no news, no social media. Read something light, stretch, or listen to calm music. Give your brain a clear 'off' signal."))

    if caf>200:
        tips.append(("☕",f"You had a lot of caffeine today ({caf}mg ≈ {caf//95} coffees)",
            f"That much caffeine is still partially active in your body right now. It won't stop you from sleeping, but it will make your sleep lighter than it should be.",
            f"Tomorrow: nothing caffeinated after {(int(u.get('wake_hour',7))+13)%24:02d}:00. Stick to water or herbal tea after that."))
    elif caf>80:
        tips.append(("☕",f"Your caffeine ({caf}mg) is reducing your sleep depth",
            "Even when caffeine doesn't keep you awake, it quietly reduces the quality of your deep sleep — so you wake up feeling less rested than you should.",
            "Switch everything to decaf or herbal tea after 1 PM. You'll notice a difference in your sleep score within 3 days."))

    if screen>90:
        tips.append(("◐",f"You had {screen} minutes of screens before bed — that's a lot",
            "That much screen time pushes back when your body naturally wants to sleep by up to 90 minutes. You're essentially tricking your brain into thinking it's still daytime.",
            "Tonight: charge your phone outside the bedroom. Use a standalone alarm clock. Your sleep will improve immediately."))
    elif screen>40:
        tips.append(("◐",f"{screen} min of screens before bed is affecting your sleep",
            "The light from screens delays the hormone that makes you sleepy. The scrolling keeps your mind active when it should be slowing down.",
            "Set a 'phone down' alarm 45 min before your target bedtime. Replace it with a book, a podcast, or just silence."))

    if ex==0:
        tips.append(("◎","You didn't exercise today — your sleep will be lighter",
            f"Without physical activity, your body isn't as tired as it needs to be for deep sleep. You took {steps:,} steps today — the target is 7,000+.",
            "Tomorrow: a 25-minute walk before 6 PM. It's the single most effective thing you can do to improve your sleep score."))
    elif steps < 5000:
        tips.append(("◎",f"Low movement today ({steps:,} steps)",
            "Even with exercise, very low daily steps means your body didn't accumulate much physical tiredness — which drives deep sleep.",
            "Walk more tomorrow: take the stairs, walk during calls, park further away. Every 1,000 steps above 5,000 improves your sleep."))

    if temp>23:
        tips.append(("◎",f"Your bedroom at {temp}°C was too warm",
            "Your body needs to cool down to fall into deep sleep. A warm room keeps you in light sleep all night — this is one of the most overlooked sleep problems.",
            "Open a window or use a fan to bring the room to 18–20°C. Keep your feet outside the covers to cool down faster."))

    if wk>=3:
        tips.append(("◑",f"You woke up {wk} times last night",
            f"Every time you wake up, your sleep resets. This is why you can sleep 8 hours and still feel tired — the sleep is fragmented.",
            "Tonight, check: is your room too warm? Is there noise or light? Did you drink too much before bed? Fix the most likely one."))

    if lat>30:
        tips.append(("◎",f"It took you {lat:.0f} minutes to fall asleep",
            f"Over 30 minutes to fall asleep usually means your mind was too active at bedtime — stress, screens, or no wind-down routine.",
            "If you're still awake after 20 minutes, get up. Do something calm and boring in dim light. Return to bed only when you feel genuinely sleepy."))

    if mh in ["anxiety","depression"] and sq<7:
        tips.append(("✦","Your mental state and sleep quality are directly connected",
            f"{'Anxiety' if 'anxiety' in mh else 'Depression'} and poor sleep feed into each other. Improving your sleep is one of the most effective ways to feel better mentally — it works both ways.",
            "Look into CBT-I (Cognitive Behavioral Therapy for Insomnia). It's the most effective long-term treatment for sleep issues linked to anxiety or depression."))

    if nap>45:
        tips.append(("◑",f"Your {nap:.0f}-minute nap is making it harder to sleep tonight",
            "A long nap reduces how tired you feel at bedtime — so you'll take longer to fall asleep and get lighter sleep tonight.",
            "Tomorrow: if you nap, keep it to 20 minutes max before 3 PM. Set an alarm before lying down."))

    if work>=12:
        tips.append(("◎",f"You worked {work:.0f} hours today",
            "After that many hours, your brain stays in 'work mode' for a long time — making it hard to fully switch off and sleep deeply.",
            "Create a clear end to your day: close everything, write tomorrow's 3 priorities on paper, then stop. No more work after that point."))

    if not tips:
        tips.append(("✦","Your habits tonight look good",
            "No major red flags in today's data. The most powerful thing you can do now is stay consistent — same bedtime and wake time every day, including weekends.",
            "Set a fixed bedtime alarm for this week. Going to bed at the same time is the single best habit for long-term sleep quality."))
        tips.append(("◎","Small changes add up over a week",
            f"You took {steps:,} steps and your stress was {stress}/10 today. These small daily habits directly affect your sleep score every night.",
            "Try to hit 7,000 steps and keep stress under 6 for 3 days in a row. Then check if your sleep score improves."))

    return tips[:6]

def ptheme(title="",h=None):
    d=dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(45,27,105,.12)",
           font=dict(color="#9d8ec7",family="Outfit"),
           title=dict(text=title,font=dict(family="Cormorant Garamond",size=16,color="#e8e0ff")),
           xaxis=dict(gridcolor="rgba(192,132,252,.08)",zeroline=False),
           yaxis=dict(gridcolor="rgba(192,132,252,.08)",zeroline=False),
           legend=dict(bgcolor="rgba(0,0,0,0)"),margin=dict(t=50,b=40,l=30,r=20))
    if h: d["height"]=h
    return d

# ═══════════════════════════════════════════════════════════
#  BOOT
# ═══════════════════════════════════════════════════════════
with st.spinner("Initializing system..."):
    df_raw = load_data()
    M = get_cached_models(df_raw)

for k,v in [("page","welcome"),("logged_in",False),("user_email",""),
            ("user_name",""),("user_profile",{}),("user_history",[])]:
    if k not in st.session_state: st.session_state[k]=v

PAGES=["home","check-in","results","tips","explore"]; LABELS=["Home","Check-In","My Results","Sleep Tips","Explore"]

st.markdown("<div style='padding:.8rem 0 .5rem'><span style='font-family:Cormorant Garamond,serif;font-size:2rem;font-weight:600;color:#e8e0ff;letter-spacing:1px'>Wi<span style='color:#c084fc'>sa</span>da 🌙</span></div>",unsafe_allow_html=True)
if st.session_state.logged_in:
    nc=st.columns(len(PAGES)+1)
    for i,(p,l) in enumerate(zip(PAGES,LABELS)):
        with nc[i]:
            if st.button(l,key=f"nb_{p}",type="primary" if st.session_state.page==p else "secondary"):
                st.session_state.page=p; st.rerun()
    with nc[-1]:
        if st.button("Sign Out",key="so"):
            for k in ["logged_in","user_email","user_name","user_profile","user_history"]: st.session_state[k]=False if k=="logged_in" else ([] if k=="user_history" else ({} if k=="user_profile" else ""))
            for k in ["result","user"]: st.session_state.pop(k,None)
            st.session_state.page="welcome"; st.rerun()
st.markdown("<div style='height:1px;background:linear-gradient(90deg,transparent,rgba(192,132,252,.22),transparent);margin:.5rem 0 1.5rem'></div>",unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════ WELCOME
if st.session_state.page=="welcome":
    st.markdown("""<div style='text-align:center;padding:3.5rem 0 2.5rem'>
      <p style='font-size:.7rem;letter-spacing:3px;color:#7c6fa0;text-transform:uppercase;margin-bottom:1rem'>Your Personal Sleep Companion</p>
      <h1 style='font-family:Cormorant Garamond,serif;font-size:4.8rem;font-weight:300;color:#e8e0ff;line-height:1.0;margin-bottom:1.8rem'>Sleep better,<br><em style='color:#c084fc'>feel better.</em></h1>
      <div style='display:flex;justify-content:center;gap:3rem;flex-wrap:wrap;max-width:700px;margin:0 auto 2.5rem'>
        <div style='display:flex;align-items:center;gap:.7rem'><span style='color:#c084fc'>◑</span><span style='color:#9d8ec7;font-size:.9rem'>Understand what's really happening while you sleep</span></div>
        <div style='display:flex;align-items:center;gap:.7rem'><span style='color:#c084fc'>◎</span><span style='color:#9d8ec7;font-size:.9rem'>Get a personalised score, bedtime, and weekly trends</span></div>
        <div style='display:flex;align-items:center;gap:.7rem'><span style='color:#c084fc'>✦</span><span style='color:#9d8ec7;font-size:.9rem'>Receive science-backed tips built around your specific habits</span></div>
      </div>
    </div>""",unsafe_allow_html=True)
    _,bc1,bc2,_=st.columns([1,1,1,1])
    with bc1:
        if st.button("✦  Create Account",use_container_width=True): st.session_state.page="register"; st.rerun()
    with bc2:
        if st.button("Sign In",use_container_width=True): st.session_state.page="login"; st.rerun()

# ═══════════════════════════════════════════════════════════ REGISTER
elif st.session_state.page=="register":
    st.markdown("<h2 style='font-family:Cormorant Garamond,serif;font-size:2.2rem;font-weight:300;color:#e8e0ff;margin-bottom:1.5rem'>Create your account</h2>",unsafe_allow_html=True)
    st.markdown("<div class='auth-wrap'>",unsafe_allow_html=True)
    name=st.text_input("Your Name"); email=st.text_input("Email Address")
    pw=st.text_input("Create a Password",type="password"); pw2=st.text_input("Confirm Password",type="password")
    st.markdown("</div>",unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    ca,cb=st.columns([1,3])
    with ca:
        if st.button("← Back"): st.session_state.page="welcome"; st.rerun()
    with cb:
        if st.button("✦  Continue →",use_container_width=True):
            if not name or not email or not pw: st.error("Please fill in all fields.")
            elif pw!=pw2: st.error("Passwords don't match.")
            else:
                st.session_state["_rn"]=name; st.session_state["_re"]=email; st.session_state["_rp"]=pw
                st.session_state.page="onboarding"; st.rerun()

# ═══════════════════════════════════════════════════════════ LOGIN
elif st.session_state.page=="login":
    st.markdown("<h2 style='font-family:Cormorant Garamond,serif;font-size:2.2rem;font-weight:300;color:#e8e0ff;margin-bottom:1.5rem'>Welcome back</h2>",unsafe_allow_html=True)
    st.markdown("<div class='auth-wrap'>",unsafe_allow_html=True)
    email=st.text_input("Email Address",key="li_e"); pw=st.text_input("Password",type="password",key="li_p")
    st.markdown("</div>",unsafe_allow_html=True)
    st.markdown("<br>",unsafe_allow_html=True)
    la,lb=st.columns([1,3])
    with la:
        if st.button("← Back",key="lb"): st.session_state.page="welcome"; st.rerun()
    with lb:
        if st.button("✦  Sign In",use_container_width=True):
            ok,name,profile,history,msg=do_login(email,pw)
            if ok:
                st.session_state.logged_in=True; st.session_state.user_email=email
                st.session_state.user_name=name; st.session_state.user_profile=profile
                st.session_state.user_history=history; st.session_state.page="home"; st.rerun()
            else: st.error(msg)

# ═══════════════════════════════════════════════════════════ ONBOARDING
elif st.session_state.page=="onboarding":
    st.markdown("<h2 style='font-family:Cormorant Garamond,serif;font-size:2.2rem;font-weight:300;color:#e8e0ff'>A little about you</h2><p style='color:#7c6fa0;margin-top:.3rem;margin-bottom:1.8rem'>This helps us personalise your analysis from day one.</p>",unsafe_allow_html=True)
    occ_opts=sorted(df_raw["occupation"].dropna().unique().tolist()) if "occupation" in df_raw.columns else []
    if "other" not in occ_opts: occ_opts=occ_opts+["other"]
    chrono_opts=sorted(df_raw["chronotype"].dropna().unique().tolist()) if "chronotype" in df_raw.columns else ["morning","evening","intermediate"]
    mh_raw=sorted(df_raw["mental_health_condition"].dropna().unique().tolist()) if "mental_health_condition" in df_raw.columns else ["none","anxiety","depression","other"]
    mh_opts=mh_raw if "none" in mh_raw else ["none"]+mh_raw
    with st.form("ob"):
        c1,c2,c3=st.columns(3)
        age=c1.number_input("Age",15,90,28); gender=c2.selectbox("Gender",["male","female"])
        bmi_v=c3.number_input("BMI",10.0,60.0,23.0,.5,help="BMI = weight(kg) ÷ height(m)²")
        c4,c5=st.columns(2)
        occ=c4.selectbox("Occupation",occ_opts)
        chrono=c5.selectbox("When are you most naturally alert and productive?",chrono_opts)
        mh=st.selectbox("Do you experience any mental health challenges? (e.g. anxiety, depression)",mh_opts)
        wd=st.slider("On weekends, how much more (or less) do you sleep vs weekdays? (hours)",-3.0,6.0,.5,.5)
        sub=st.form_submit_button("✦  Create My Account")
    if sub:
        profile=dict(age=age,gender=gender,bmi=bmi_v,occupation=occ,chronotype=chrono,mental_health_condition=mh,weekend_sleep_diff_hrs=wd)
        ok,msg=register(st.session_state["_rn"],st.session_state["_re"],st.session_state["_rp"],profile)
        if ok:
            st.session_state.logged_in=True; st.session_state.user_email=st.session_state["_re"]
            st.session_state.user_name=st.session_state["_rn"]; st.session_state.user_profile=profile
            st.session_state.user_history=[]; st.session_state.page="home"; st.rerun()
        else: st.error(msg)

# ═══════════════════════════════════════════════════════════ HOME
elif st.session_state.page=="home" and st.session_state.logged_in:
    name=st.session_state.user_name
    _today = datetime.date.today().strftime("%A, %B %d").replace(" 0", " ")
    st.markdown(f"""<div style='padding:.5rem 0 1.5rem'>
      <p style='font-size:.72rem;letter-spacing:3px;color:#7c6fa0;text-transform:uppercase'>{_today}</p>
      <h1 style='font-family:Cormorant Garamond,serif;font-size:3rem;font-weight:300;color:#e8e0ff;margin-top:.3rem'>
        Good evening, <em style='color:#c084fc'>{name}</em> 🌙</h1>
      <p style='font-size:.95rem;color:#7c6fa0;margin-top:.6rem'>Ready to analyse tonight's sleep?</p>
    </div>""",unsafe_allow_html=True)

    ha,hb=st.columns([1,1],gap="large")
    with ha:
        if st.button("✦  Analyse Tonight's Sleep",use_container_width=True): st.session_state.page="check-in"; st.rerun()
    with hb:
        hist=st.session_state.get("user_history",[])
        if hist:
            last=hist[-1]; lbl,lcol=score_label(last["score"])
            st.markdown(f"""<div class='card-gold'>
              <div style='font-size:.65rem;letter-spacing:2px;color:#9d8ec7;text-transform:uppercase;margin-bottom:.4rem'>Last Entry · {last.get("date","—")}</div>
              <div style='display:flex;align-items:center;gap:1.5rem'>
                <div><div style='font-family:Cormorant Garamond,serif;font-size:3.5rem;color:{lcol};font-weight:300;line-height:1'>{last["score"]}</div>
                <div style='font-family:Cormorant Garamond,serif;color:{lcol};font-style:italic'>{lbl}</div></div>
                <div style='flex:1'>
                  <div style='font-size:.8rem;color:#9d8ec7;margin-bottom:.3rem'>Sleep Quality: <span style='color:#e8e0ff'>{last["sq"]:.1f}/10</span></div>
                  <div style='font-size:.8rem;color:#9d8ec7'>Ideal Bedtime: <span style='color:#f4c97a'>{last["bedtime"]}</span></div>
                </div>
              </div>
            </div>""",unsafe_allow_html=True)

    # Weekly history chart
    hist=st.session_state.get("user_history",[])
    if len(hist)>=2:
        st.markdown("<div class='divider'></div>",unsafe_allow_html=True)
        st.markdown("<p style='font-family:Cormorant Garamond,serif;font-size:1.4rem;color:#e8e0ff;margin-bottom:1rem'>Your Sleep This Week</p>",unsafe_allow_html=True)
        recent=hist[-7:]
        dates=[e.get("date","") for e in recent]
        scores=[e.get("score",0) for e in recent]
        sqs=[e.get("sq",0) for e in recent]
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=dates,y=scores,mode="lines+markers",name="Sleep Score",
            line=dict(color="#f4c97a",width=2.5),marker=dict(size=8,color="#f4c97a")))
        fig.add_trace(go.Scatter(x=dates,y=[s*10 for s in sqs],mode="lines+markers",name="Quality ×10",
            line=dict(color="#c084fc",width=2,dash="dot"),marker=dict(size=6,color="#c084fc")))
        fig.add_hline(y=70,line_dash="dash",line_color="rgba(167,139,250,.3)",annotation_text="Good threshold")
        lo=ptheme("Weekly Sleep Score Trend",h=260); lo["margin"]=dict(t=40,b=30,l=30,r=20)
        lo["yaxis"]["range"]=[0,105]
        fig.update_layout(**lo)
        st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        # day tiles
        cols=st.columns(len(recent))
        for col,e in zip(cols,recent):
            lbl,lcol=score_label(e["score"])
            with col:
                st.markdown(f"""<div class='week-day'>
                  <div class='week-date'>{e.get("date","")[:5]}</div>
                  <div class='week-score' style='color:{lcol}'>{e["score"]}</div>
                  <div style='font-size:.65rem;color:#6d5fa0;margin-top:.2rem'>{lbl}</div>
                </div>""",unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════ CHECK-IN
elif st.session_state.page=="check-in" and st.session_state.logged_in:
    _today=datetime.date.today()
    _month=_today.month
    _season=("winter" if _month in [12,1,2] else "spring" if _month in [3,4,5] else "summer" if _month in [6,7,8] else "autumn")
    _day_type="weekend" if _today.weekday()>=5 else "weekday"
    _date_str = _today.strftime("%A, %B %d, %Y").replace(" 0", " ")

    st.markdown(f"""<h2 style='font-family:Cormorant Garamond,serif;font-size:2.4rem;font-weight:300;color:#e8e0ff'>Tonight's Check-In</h2>
    <p style='color:#7c6fa0;font-size:1.1rem;font-weight:500;margin-top:.3rem;margin-bottom:1.8rem'>{_date_str}</p>""",unsafe_allow_html=True)

    with st.form("ci"):
        st.markdown("<div class='form-section'>Your Sleep</div>",unsafe_allow_html=True)
        st.markdown("<div class='hint-box'>Tell us about last night's sleep.</div>",unsafe_allow_html=True)
        c1,c2=st.columns(2)
        sl_hrs=c1.slider("How many hours did you sleep?",1.0,16.0,7.0,.5)
        sl_dep=c2.slider("How deep was your sleep? (1 = barely · 10 = very deep)",1,10,6)
        c3,c4=st.columns(2)
        sl_lat=c3.slider("How long to fall asleep? (minutes)",0,120,15)
        wk_ep=c4.slider("Times woken up at night",0,15,1)
        c5,c6=st.columns(2)
        nap=c5.slider("Nap today (minutes)",0,180,0)
        r_temp=c6.slider("Bedroom temperature (°C)",14,32,20)
        st.markdown("<p style='font-size:.78rem;color:#9d8ec7;letter-spacing:.8px;margin:.6rem 0 .3rem'>What time do you wake up?</p>",unsafe_allow_html=True)
        wt1,wt2=st.columns(2)
        w_hr=wt1.selectbox("Hour",list(range(0,24)),index=7,format_func=lambda x:f"{x:02d}")
        w_min=wt2.selectbox("Minute",[0,15,30,45],index=0,format_func=lambda x:f"{x:02d}")
        s1,s2=st.columns(2)
        sl_aid=s1.selectbox("Did you take anything to help you sleep last night?",["No","Yes"])
        shft=s2.selectbox("Did you work a night shift?",["No","Yes"])
        deep_approx=int(sl_dep*3.5); rem_approx=int(sl_dep*2.5)

        st.markdown("<div class='divider'></div>",unsafe_allow_html=True)
        st.markdown("<div class='form-section'>Your Day</div>",unsafe_allow_html=True)
        d1,d2,d3=st.columns(3)
        stress=d1.slider("Stress level (1 = calm · 10 = very stressed)",1,10,4)
        work_h=d2.slider("Hours worked today",0,16,8)
        exercised=d3.selectbox("Did you exercise today?",["Yes","No"])
        d4,d5=st.columns(2)
        steps=d4.slider("Steps today (approx.)",0,25000,7000,500)
        screen=d5.slider("Screen time before bed (minutes)",0,240,45)
        st.markdown("""<div class='hint-box' style='margin-top:.6rem'><strong style='color:#e8e0ff'>Caffeine reference</strong><br>
          Espresso = 63 mg &nbsp;·&nbsp; Drip coffee (240ml) = 95 mg &nbsp;·&nbsp; Latte / Cappuccino = 63 mg &nbsp;·&nbsp;
          Energy drink (250ml) = 80 mg &nbsp;·&nbsp; Green tea = 28 mg &nbsp;·&nbsp; Black tea = 47 mg</div>""",unsafe_allow_html=True)
        caf=st.slider("Total caffeine today (mg)",0,800,95)
        st.markdown("<br>",unsafe_allow_html=True)
        submitted=st.form_submit_button("✦  Analyse My Sleep Tonight")

    if submitted:
        prof=st.session_state.get("user_profile",{}); fm=M["maxes"]
        user=dict(
            age=float(prof.get("age",30)),gender=str(prof.get("gender","male")),
            bmi=float(prof.get("bmi",23)),occupation=str(prof.get("occupation","other")),
            chronotype=str(prof.get("chronotype","intermediate")),
            mental_health_condition=str(prof.get("mental_health_condition","none")),
            weekend_sleep_diff_hrs=float(prof.get("weekend_sleep_diff_hrs",.5)),
            sleep_duration_hrs=float(sl_hrs),sleep_latency_mins=float(sl_lat),
            wake_episodes_per_night=float(wk_ep),rem_percentage=float(rem_approx),
            deep_sleep_percentage=float(deep_approx),nap_duration_mins=float(nap),
            wake_hour=w_hr,wake_minute=w_min,stress_score=float(stress),
            exercise_day=1 if exercised=="Yes" else 0,work_hours_that_day=float(work_h),
            heart_rate_resting_bpm=70.0,caffeine_mg_before_bed=float(caf),
            alcohol_units_before_bed=0.0,screen_time_before_bed_mins=float(screen),
            room_temperature_celsius=float(r_temp),steps_that_day=float(steps),
            sleep_aid_used=1 if sl_aid=="Yes" else 0,shift_work=1 if shft=="Yes" else 0,
            season=_season,day_type=_day_type,
            restorative_sleep_score=float(rem_approx+deep_approx),
            pre_bed_stimulant_load=float(caf/(fm["caffeine_mg_before_bed"]+1)),
            sleep_debt_proxy=float(abs(prof.get("weekend_sleep_diff_hrs",.5))),
            sleep_disruption_score=float(stress/(fm["stress_score"]+1)+wk_ep/(fm["wake_episodes_per_night"]+1)+sl_lat/(fm["sleep_latency_mins"]+1)),
        )
        result=predict(user,M,df_raw)
        st.session_state["result"]=result; st.session_state["user"]=user
        entry={"date":_today.strftime("%b %d").replace(" 0", " "),"score":result["score"],"sq":result["sleep_quality"],"bedtime":result["bedtime"],"cog":result["cognitive"]}
        st.session_state.user_history.append(entry)
        save_entry(st.session_state.user_email,entry)
        st.session_state.page="results"; st.rerun()

# ═══════════════════════════════════════════════════════════ RESULTS
# ═══════════════════════════════════════════════════════════ RESULTS
elif st.session_state.page == "results" and st.session_state.logged_in:
    if "result" not in st.session_state or not st.session_state.get("result"):
        st.info("Complete your check-in first.")
        if st.button("Go to Check-In"): st.session_state.page = "check-in"; st.rerun()
    else:
        p = st.session_state["result"];
        u = st.session_state["user"]
        sq = p["sleep_quality"];
        cog = p["cognitive"]
        sq_lbl, sq_col = sq_label(sq);
        cog_lbl, cog_col = cog_label(cog)
        score_lbl, score_col = score_label(p["score"])

        # ── FIXED DISORDER LOGIC ──
        dis = p["disorder"]
        is_ok = dis.lower() in ["none", "no disorder", "healthy", "normal"]

        # Dynamic color coding based on severity
        if is_ok:
            dis_col = "#a78bfa"  # Purple (Good)
        elif "mild" in dis.lower() or "insomnia" in dis.lower():
            dis_col = "#f4c97a"  # Orange/Yellow (Warning)
        else:
            dis_col = "#f87171"  # Red (Severe/Apnea)

        _today_str = datetime.date.today().strftime("%A, %B %d").replace(" 0", " ")

        st.markdown(
            f"""<p style='font-size:.72rem;letter-spacing:3px;color:#7c6fa0;text-transform:uppercase'>{_today_str} · Sleep Report</p>
        <h2 style='font-family:Cormorant Garamond,serif;font-size:2.4rem;font-weight:300;color:#e8e0ff;margin-top:.3rem;margin-bottom:1.5rem'>Your Results</h2>""",
            unsafe_allow_html=True)


        # ── 4 metric cards — full width row ──────────────────────────────
        def mk(label, num_html, word, word_col, bar_pct, pts):
            bar_col = word_col
            bullets = "".join(f"<li>{pt}</li>" for pt in pts)
            return f"""<div style='padding:1.6rem 1.4rem;border-radius:22px;background:rgba(255,255,255,.03);
                border:1px solid rgba(192,132,252,.13);height:100%'>
              <div style='font-size:.6rem;letter-spacing:2.5px;color:#6d5fa0;text-transform:uppercase;
                   font-weight:600;margin-bottom:.5rem'>{label}</div>
              <div style='font-family:Cormorant Garamond,serif;font-size:2.8rem;font-weight:300;
                   color:{word_col};line-height:1'>{num_html}</div>
              <div style='font-family:Cormorant Garamond,serif;font-style:italic;font-size:.9rem;
                   color:{word_col};margin:.3rem 0 .6rem'>{word}</div>
              <div style='background:rgba(255,255,255,.06);border-radius:99px;height:4px;
                   overflow:hidden;margin-bottom:.8rem'>
                <div style='height:100%;width:{bar_pct}%;background:{bar_col};border-radius:99px'></div>
              </div>
              <ul style='list-style:none;padding:0;margin:0;border-top:1px solid rgba(192,132,252,.1);
                   padding-top:.6rem'>{bullets}</ul>
            </div>"""


        r_col2 = "#7dd3a8" if p["felt_rested"] else "#f87171"
        dis_show2 = "Healthy" if is_ok else dis.replace("_", " ").title()
        hrs = u.get("sleep_duration_hrs", 7);
        lat = u.get("sleep_latency_mins", 15)
        wk = u.get("wake_episodes_per_night", 1);
        deep = u.get("deep_sleep_percentage", 18)
        stress = u.get("stress_score", 5);
        caf = u.get("caffeine_mg_before_bed", 0)
        screen = u.get("screen_time_before_bed_mins", 0);
        temp = u.get("room_temperature_celsius", 20)

        sq_pts = []
        if sq >= 7:
            sq_pts.append("You slept well — your body got the recovery it needed")
        elif sq >= 5:
            sq_pts.append("Okay sleep, but not fully restorative")
        else:
            sq_pts.append("Poor sleep — your body didn't fully recover last night")
        if hrs < 6: sq_pts.append(f"Only {hrs:.1f} hrs slept — you need at least 7 for full recovery")
        if lat > 25: sq_pts.append(f"Took {lat:.0f} min to fall asleep — mind was active at bedtime")
        if wk >= 2: sq_pts.append(f"Woke up {wk} times — each interruption cuts deep sleep short")
        if deep < 14: sq_pts.append("Low deep sleep — less physical repair happened overnight")
        if not sq_pts[1:]: sq_pts.append(f"Slept {hrs:.1f} hrs with {wk} waking(s)")

        cog_pts = []
        if cog >= 75:
            cog_pts.append("Brain is well-rested — sharp focus expected today")
        elif cog >= 60:
            cog_pts.append("Decent focus, may feel slower in the afternoon")
        else:
            cog_pts.append("Focus will be harder than usual today — take more breaks")
        if hrs < 6: cog_pts.append(f"Only {hrs:.1f} hrs sleep reduces memory and reaction speed")
        if cog < 60: cog_pts.append("A 20-min nap between 1–3 PM can help restore focus")

        rest_pts = []
        if p["felt_rested"]:
            rest_pts = ["You should feel refreshed when you wake up",
                        "Your sleep quality was enough for morning energy"]
        else:
            if hrs < 5.5: rest_pts.append(f"Only {hrs:.1f} hrs — not enough time for your body to rest")
            if wk >= 3: rest_pts.append(f"Waking {wk} times prevented reaching deep rest")
            rest_pts.append("Expect 20–30 min of grogginess after waking — that's normal")

        # ── NEW FIXED DISORDER TEXT SELECTION ──
        dis_pts = []
        np2 = p["none_prob"];
        conf = int(np2 * 100)

        if is_ok:
            dis_pts.append(f"No sleep disorder signs — your pattern looks normal ({conf}% confidence)")
            dis_pts.append("Keep consistent sleep times to maintain this healthy baseline")

        elif "insomnia" in dis.lower():
            dis_pts.append("Your pattern shows insomnia signs — trouble falling or staying asleep")
            if lat > 25: dis_pts.append(f"Taking {lat:.0f} min to fall asleep is above the normal 10–20 min")
            dis_pts.append("A consistent pre-bed routine and fixed wake time helps significantly")

        elif "apnea" in dis.lower():
            dis_pts.append("Your profile shows sleep apnea markers")
            dis_pts.append(f"Waking {wk} times and unrefreshing sleep are key signs")
            dis_pts.append("Worth visiting a doctor — sleep apnea is very treatable")

        elif "mild" in dis.lower():
            dis_pts.append("Your sleep shows mild signs of disruption")
            dis_pts.append("Consistent habits for 2–3 weeks often fixes most of these issues")

        elif "severe" in dis.lower():
            dis_pts.append("Your profile shows severe sleep disruption markers")
            dis_pts.append("Consider reviewing your daily habits or consulting a sleep specialist")

        else:
            dis_pts.append(f"Risk category identified: {dis.title()}")
            dis_pts.append("Monitor your stress levels and wake episodes closely this week")

        c1, c2, c3, c4 = st.columns(4, gap="small")
        with c1:
            st.markdown(
                mk("Sleep Quality", f"{sq:.1f}<span style='font-size:1rem;color:#6d5fa0'>/10</span>", sq_lbl, sq_col,
                   int(sq / 10 * 100), ["".join([
                                                    f"<li style='font-size:.79rem;color:#9d8ec7;line-height:1.5;padding:.22rem 0;display:flex;gap:.45rem;align-items:flex-start'><span style='color:#c084fc;font-size:.6rem;margin-top:.2rem;flex-shrink:0'>◑</span>" + pt + "</li>"
                                                    for pt in sq_pts])]), unsafe_allow_html=True)
        with c2:
            st.markdown(
                mk("Focus Tomorrow", f"{cog:.0f}<span style='font-size:1rem;color:#6d5fa0'>/100</span>", cog_lbl,
                   cog_col, int(cog), ["".join([
                                                   f"<li style='font-size:.79rem;color:#9d8ec7;line-height:1.5;padding:.22rem 0;display:flex;gap:.45rem;align-items:flex-start'><span style='color:#c084fc;font-size:.6rem;margin-top:.2rem;flex-shrink:0'>◑</span>" + pt + "</li>"
                                                   for pt in cog_pts])]), unsafe_allow_html=True)
        with c3:
            st.markdown(mk("Morning Feeling", "Rested" if p["felt_rested"] else "Tired", "", r_col2,
                           100 if p["felt_rested"] else 30, ["".join([
                                                                         f"<li style='font-size:.79rem;color:#9d8ec7;line-height:1.5;padding:.22rem 0;display:flex;gap:.45rem;align-items:flex-start'><span style='color:#c084fc;font-size:.6rem;margin-top:.2rem;flex-shrink:0'>◑</span>" + pt + "</li>"
                                                                         for pt in rest_pts])]), unsafe_allow_html=True)
        with c4:
            st.markdown(mk("Sleep Health", dis_show2, "", dis_col, int(np2 * 100), ["".join([
                                                                                                f"<li style='font-size:.79rem;color:#9d8ec7;line-height:1.5;padding:.22rem 0;display:flex;gap:.45rem;align-items:flex-start'><span style='color:#c084fc;font-size:.6rem;margin-top:.2rem;flex-shrink:0'>◑</span>" + pt + "</li>"
                                                                                                for pt in dis_pts])]),
                        unsafe_allow_html=True)

        # ── Overall score bar ─────────────────────────────────────────────
        st.markdown(f"""<div style='background:rgba(255,255,255,.03);border:1px solid rgba(192,132,252,.12);
            border-radius:16px;padding:1.1rem 1.8rem;margin:.6rem 0;display:flex;align-items:center;gap:2rem'>
          <div style='flex-shrink:0;min-width:80px'>
            <div style='font-size:.58rem;letter-spacing:2px;color:#6d5fa0;text-transform:uppercase;margin-bottom:.2rem'>Overall</div>
            <div style='font-family:Cormorant Garamond,serif;font-size:2.8rem;color:{score_col};font-weight:300;line-height:1'>{p["score"]}</div>
            <div style='font-family:Cormorant Garamond,serif;color:{score_col};font-style:italic;font-size:.88rem'>{score_lbl}</div>
          </div>
          <div style='flex:1'>
            <div style='background:rgba(255,255,255,.06);border-radius:99px;height:7px;overflow:hidden'>
              <div style='height:100%;width:{p["score"]}%;background:linear-gradient(90deg,{score_col}66,{score_col});border-radius:99px'></div>
            </div>
            <div style='display:flex;justify-content:space-between;margin-top:.35rem;font-size:.68rem;color:#6d5fa0'>
              <span>Needs Care</span><span style='margin-left:18%'>Fair</span><span style='margin-left:15%'>Good</span><span>Excellent</span>
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Bedtime + What hurt your sleep ────────────────────────────────
        bc, sc = st.columns([1, 1.3], gap="large")
        with bc:
            sq_note = "Good sleep quality — 7 hours is your sweet spot tonight" if sq >= 7 else "Okay sleep quality — 7.5 hrs helps compensate" if sq >= 5 else "Rough night — 8.5 hrs gives your body more recovery time"
            st.markdown(f"""<div style='text-align:center;background:linear-gradient(160deg,rgba(244,201,122,.12),rgba(192,132,252,.07));
                border:1px solid rgba(244,201,122,.28);border-radius:24px;padding:2rem 1.5rem'>
              <div style='font-size:.62rem;letter-spacing:2.5px;color:#9d8ec7;text-transform:uppercase;margin-bottom:.5rem'>Ideal Bedtime Tonight</div>
              <div style='font-family:Cormorant Garamond,serif;font-size:5rem;color:#f4c97a;font-weight:300;line-height:1;letter-spacing:3px'>{p["bedtime"]}</div>
              <div style='font-size:.82rem;color:#c084fc;margin-top:.3rem'>{p["ideal_hrs"]:.1f} hours of sleep recommended</div>
              <ul style='list-style:none;padding:0;margin:.9rem 0 0;text-align:left'>
                <li style='font-size:.81rem;color:#9d8ec7;padding:.28rem 0;display:flex;gap:.6rem;align-items:flex-start'><span style='color:#f4c97a;font-size:.6rem;margin-top:.22rem;flex-shrink:0'>◎</span>{sq_note}</li>
                <li style='font-size:.81rem;color:#9d8ec7;padding:.28rem 0;display:flex;gap:.6rem;align-items:flex-start'><span style='color:#f4c97a;font-size:.6rem;margin-top:.22rem;flex-shrink:0'>◎</span>Set a reminder 30 min before {p["bedtime"]} to start winding down</li>
                <li style='font-size:.81rem;color:#9d8ec7;padding:.28rem 0;display:flex;gap:.6rem;align-items:flex-start'><span style='color:#f4c97a;font-size:.6rem;margin-top:.22rem;flex-shrink:0'>◎</span>Same bedtime every night is the most powerful sleep habit you can build</li>
              </ul>
            </div>""", unsafe_allow_html=True)

        with sc:
            st.markdown(
                "<p style='font-family:Cormorant Garamond,serif;font-size:1.3rem;color:#e8e0ff;margin-bottom:.3rem'>What hurt your sleep tonight?</p><p style='font-size:.8rem;color:#7c6fa0;margin-bottom:.9rem'>The habits from today that had the biggest impact on your sleep quality.</p>",
                unsafe_allow_html=True)
            if p.get("saboteurs"):
                for sab in p["saboteurs"]:
                    f1 = sab["features"][0]
                    if f1 in SAB_MAP:
                        sab_name, sab_pts, sab_act = SAB_MAP[f1]
                        pts_li = "".join(
                            f"<li style='font-size:.8rem;color:#9d8ec7;line-height:1.5;padding:.22rem 0;display:flex;gap:.5rem;align-items:flex-start'><span style='color:#f87171;font-size:.6rem;margin-top:.2rem;flex-shrink:0'>◑</span>{pt}</li>"
                            for pt in sab_pts)
                        st.markdown(f"""<div style='padding:1.3rem 1.5rem;border-radius:18px;background:rgba(248,113,113,.05);
                            border:1px solid rgba(248,113,113,.2);margin-bottom:.8rem'>
                          <div style='font-size:.6rem;letter-spacing:2px;color:#f87171;text-transform:uppercase;margin-bottom:.35rem'>Biggest factor tonight</div>
                          <div style='font-size:1.05rem;color:#fca5a5;font-weight:600;margin-bottom:.45rem'>{sab_name}</div>
                          <ul style='list-style:none;padding:0;margin:0 0 .55rem'>{pts_li}</ul>
                          <div style='font-size:.79rem;color:#c084fc;border-top:1px solid rgba(248,113,113,.15);padding-top:.5rem'><strong>Fix it tonight:</strong> {sab_act}</div>
                        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

        # ── Tips 2-column ─────────────────────────────────────────────────
        st.markdown(
            "<p style='font-family:Cormorant Garamond,serif;font-size:1.6rem;color:#e8e0ff;margin-bottom:.3rem'>Your Sleep Tips Tonight</p><p style='font-size:.82rem;color:#7c6fa0;margin-bottom:1.2rem'>Based on your data today — written for you, not a generic list.</p>",
            unsafe_allow_html=True)
        tips = get_personalized_tips(u, p)
        tc1, tc2 = st.columns(2, gap="medium")
        for i, (sym, title, body, action) in enumerate(tips):
            col = tc1 if i % 2 == 0 else tc2
            with col:
                st.markdown(f"""<div style='display:flex;gap:1rem;align-items:flex-start;padding:1.3rem 1.4rem;
                    border-radius:18px;background:rgba(192,132,252,.05);border:1px solid rgba(192,132,252,.13);
                    margin-bottom:.8rem'>
                  <div style='font-size:1.3rem;flex-shrink:0;padding-top:1px'>{sym}</div>
                  <div>
                    <div style='font-size:.91rem;color:#e8e0ff;font-weight:600;margin-bottom:.28rem'>{title}</div>
                    <div style='font-size:.81rem;color:#9d8ec7;line-height:1.6;margin-bottom:.45rem'>{body}</div>
                    <div style='display:inline-block;font-size:.74rem;letter-spacing:.5px;color:#c084fc;
                         border:1px solid rgba(192,132,252,.28);border-radius:99px;padding:3px 12px'>{action}</div>
                  </div>
                </div>""", unsafe_allow_html=True)

        # ── Daily stats strip ─────────────────────────────────────────────
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        hist_all = st.session_state.get("user_history", [])
        n_nights = len(hist_all)
        n_good = len([e for e in hist_all if e.get("score", 0) >= 60])
        avg_sc = round(sum(e.get("score", 0) for e in hist_all) / max(1, n_nights))
        best_sc = max((e.get("score", 0) for e in hist_all), default=0)
        s1, s2, s3, s4 = st.columns(4, gap="small")
        for col, val, lbl, clr in [(s1, n_nights, "Nights tracked", "#c084fc"),
                                   (s2, n_good, "Good nights (60+)", "#7dd3a8"),
                                   (s3, avg_sc, "Your avg score", "#f4c97a"),
                                   (s4, best_sc, "Your best score", "#a78bfa")]:
            with col:
                st.markdown(f"""<div style='padding:1.2rem 1rem;border-radius:18px;background:rgba(255,255,255,.03);
                    border:1px solid rgba(192,132,252,.1);text-align:center'>
                  <div style='font-size:.58rem;letter-spacing:2px;color:#6d5fa0;text-transform:uppercase;margin-bottom:.35rem'>{lbl}</div>
                  <div style='font-family:Cormorant Garamond,serif;font-size:2.5rem;color:{clr};font-weight:300'>{val}</div>
                </div>""", unsafe_allow_html=True)
        st.markdown("""<div style='background:rgba(124,58,237,.08);border:1px solid rgba(192,132,252,.2);
            border-radius:16px;padding:1rem 1.5rem;margin-top:.6rem;display:flex;align-items:center;justify-content:space-between;gap:1rem'>
          <div>
            <div style='font-size:.85rem;color:#c084fc;font-weight:500;margin-bottom:.2rem'>Come back tomorrow night</div>
            <div style='font-size:.79rem;color:#9d8ec7'>The more nights you log, the clearer your patterns — and the smarter your tips get.</div>
          </div>
          <div style='font-size:.75rem;color:#7c6fa0;flex-shrink:0;text-align:right'>Your streak matters.<br>Every night counts.</div>
        </div>""", unsafe_allow_html=True)
# ═══════════════════════════════════════════════════════════ TIPS (standalone)
elif st.session_state.page=="tips" and st.session_state.logged_in:
    if "result" not in st.session_state or not st.session_state.get("result"):
        st.info("Complete your check-in first.")
        if st.button("Go to Check-In"): st.session_state.page="check-in"; st.rerun()
    else:
        p=st.session_state["result"]; u=st.session_state["user"]; lbl,lcol=score_label(p["score"])
        st.markdown("<p style='font-size:.72rem;letter-spacing:3px;color:#7c6fa0;text-transform:uppercase'>Based on your check-in</p><h2 style='font-family:Cormorant Garamond,serif;font-size:2.4rem;font-weight:300;color:#e8e0ff;margin-top:.3rem;margin-bottom:1.8rem'>Your Sleep Tips</h2>",unsafe_allow_html=True)
        tips=get_personalized_tips(u,p); t1,t2=st.columns([1.6,1],gap="large")
        with t1:
            for sym,title,body,action in tips:
                st.markdown(f"""<div class='tip-row'><div class='tip-icon'>{sym}</div>
                  <div><div class='tip-title'>{title}</div><div class='tip-body'>{body}</div>
                  <div class='tip-action'>Tonight: {action}</div></div></div>""",unsafe_allow_html=True)
        with t2:
            st.markdown(f"""<div class='bedtime-hero' style='margin-bottom:1rem'>
              <div class='bt-tag'>Go to Bed At</div><div class='bt-clock'>{p["bedtime"]}</div>
              <div style='font-size:.82rem;color:#9d8ec7;margin-top:.6rem'>For {p["ideal_hrs"]:.1f} hrs · Score {p["score"]}/100</div>
            </div>
            <div class='card' style='text-align:center;margin-top:.9rem'>
              <div style='font-size:.62rem;letter-spacing:2px;color:#6d5fa0;text-transform:uppercase;margin-bottom:.5rem'>Sleep Score</div>
              <div style='font-family:Cormorant Garamond,serif;font-size:3.5rem;color:{lcol};font-weight:300'>{p["score"]}</div>
              <div style='font-family:Cormorant Garamond,serif;color:{lcol};font-style:italic;font-size:1.1rem'>{lbl}</div>
            </div>""",unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════ EXPLORE
elif st.session_state.page=="explore" and st.session_state.logged_in:
    df=df_raw.copy()
    st.markdown("""<p style='font-size:.72rem;letter-spacing:3px;color:#7c6fa0;text-transform:uppercase'>Sleep Science</p>
    <h2 style='font-family:Cormorant Garamond,serif;font-size:2.4rem;font-weight:300;color:#e8e0ff;margin-top:.3rem;margin-bottom:.4rem'>Understanding Sleep</h2>
    <p style='font-size:.9rem;color:#7c6fa0;margin-bottom:2rem'>A science-backed guide to why you sleep, what happens when you don't, and what the data actually shows.</p>""",unsafe_allow_html=True)

    t1,t2,t3=st.tabs(["  Why Sleep Matters  ","  Your Habits & Sleep  ","  Sleep Disorders  "])

    with t1:
        st.markdown("""<div class='article-section'>
          <div class='article-heading'>Sleep is not downtime — it's active restoration</div>
          <div class='article-body'>Every night, your brain cycles through four sleep stages — N1 (light), N2 (intermediate), N3 (deep/slow-wave), and REM. Each full cycle takes approximately 90 minutes, and you need 4–6 complete cycles for full restoration. Cutting sleep short doesn't just reduce quantity — it disproportionately removes the final cycles where REM is concentrated.</div>
        </div>""",unsafe_allow_html=True)
        st.markdown('<span class="article-stat">REM sleep = emotional regulation</span><span class="article-stat">Deep sleep = physical repair</span><span class="article-stat">N2 = memory consolidation</span>',unsafe_allow_html=True)
        st.markdown("<br>",unsafe_allow_html=True)

        c1,c2=st.columns(2)
        with c1:
            fig=go.Figure(go.Histogram(x=df["sleep_duration_hrs"],nbinsx=35,marker_color="rgba(192,132,252,.75)",opacity=.9))
            fig.add_vline(x=7,line_dash="dash",line_color="#f4c97a",annotation_text="7h minimum",annotation_font_color="#f4c97a")
            fig.update_layout(**ptheme("How Long People Actually Sleep",260)); st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        with c2:
            age_d=df.groupby("age")["deep_sleep_percentage"].mean().reset_index()
            fig=go.Figure(go.Scatter(x=age_d["age"],y=age_d["deep_sleep_percentage"],mode="lines+markers",
                line=dict(color="rgba(167,139,250,.9)",width=2.5),marker=dict(size=5,color="#a78bfa"),
                fill="tozeroy",fillcolor="rgba(124,58,237,.08)"))
            fig.update_layout(**ptheme("Deep Sleep Declines With Age",260)); st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

        st.markdown("""<div class='article-section' style='margin-top:1.5rem'>
          <div class='article-heading'>What happens when you're sleep-deprived</div>
          <div class='article-body'>After 17 hours awake, cognitive impairment is equivalent to a blood alcohol level of 0.05%. After 24 hours, it reaches 0.10% — legally drunk in most countries. But unlike being drunk, sleep deprivation doesn't feel as impairing as it actually is. Your self-assessment of performance stays higher than your actual performance — which makes it dangerous. Chronic sleep restriction (6 hours/night for two weeks) produces the same cognitive deficit as 48 hours of total sleep deprivation. And you don't fully recover in one night — it takes 4–5 nights of full sleep to restore cognitive function.</div>
        </div>""",unsafe_allow_html=True)

        c3,c4=st.columns(2)
        with c3:
            samp=df.sample(min(2000,len(df)),random_state=3)
            fig=go.Figure(go.Scatter(x=samp["sleep_duration_hrs"],y=samp["cognitive_performance_score"],
                mode="markers",marker=dict(color=samp["sleep_quality_score"],colorscale="Purp",size=4,opacity=.55,showscale=True,
                colorbar=dict(title=dict(text="Sleep Quality",font=dict(color="#9d8ec7")),tickfont=dict(color="#9d8ec7")))))
            fig.update_layout(**ptheme("Sleep Duration → Cognitive Score",280)); st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        with c4:
            felt=df["felt_rested"].value_counts()
            fig=go.Figure(go.Pie(values=felt.values,labels=["Felt Rested" if l==1 else "Did Not Feel Rested" for l in felt.index],
                hole=.58,marker_colors=["rgba(167,139,250,.85)","rgba(248,113,113,.6)"],
                textfont=dict(color="#e8e0ff")))
            fig.update_layout(**ptheme("How Rested People Actually Feel",280)); st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

        pct_under7=round((df["sleep_duration_hrs"]<7).mean()*100,1)
        st.markdown(f"""<div class='card-purple'>
          <div style='font-size:.65rem;letter-spacing:2px;color:#c084fc;text-transform:uppercase;margin-bottom:.6rem'>Key Finding From Our Data</div>
          <div style='font-size:.9rem;color:#d4c8f5;line-height:1.9'><strong style='color:#e8e0ff'>{pct_under7}% of people sleep under 7 hours.</strong> The average in our dataset is {df["sleep_duration_hrs"].mean():.1f} hours — below the evidence-based minimum of 7 hours for adults. Among those sleeping under 6 hours, cognitive performance scores average {df[df["sleep_duration_hrs"]<6]["cognitive_performance_score"].mean():.0f}/100, compared to {df[df["sleep_duration_hrs"]>=7]["cognitive_performance_score"].mean():.0f}/100 for those getting 7+ hours. That's a {abs(df[df["sleep_duration_hrs"]<6]["cognitive_performance_score"].mean()-df[df["sleep_duration_hrs"]>=7]["cognitive_performance_score"].mean()):.0f}-point difference in daily cognitive function.</div>
        </div>""",unsafe_allow_html=True)

    with t2:
        st.markdown("""<div class='article-section'>
          <div class='article-heading'>Stress is the #1 enemy of deep sleep</div>
          <div class='article-body'>Stress activates the hypothalamic-pituitary-adrenal (HPA) axis, releasing cortisol. Cortisol and melatonin have an inverse relationship — when one rises, the other falls. Evening stress keeps cortisol elevated precisely when your brain needs it low to produce melatonin and initiate sleep. High-stress individuals spend more time in lighter sleep stages and less time in the restorative N3 and REM phases.</div>
        </div>""",unsafe_allow_html=True)

        c1,c2=st.columns(2)
        with c1:
            samp=df.sample(min(2000,len(df)),random_state=4)
            fig=go.Figure(go.Scatter(x=samp["stress_score"],y=samp["sleep_quality_score"],
                mode="markers",marker=dict(color=samp["stress_score"],colorscale="RdPu",size=4,opacity=.55)))
            # trend line approximation
            z=np.polyfit(samp["stress_score"],samp["sleep_quality_score"],1)
            xr=np.linspace(samp["stress_score"].min(),samp["stress_score"].max(),100)
            fig.add_trace(go.Scatter(x=xr,y=np.polyval(z,xr),mode="lines",
                line=dict(color="#f4c97a",width=2,dash="dot"),name="Trend",showlegend=False))
            fig.update_layout(**ptheme("Stress vs Sleep Quality — Every Point is a Person",280)); st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        with c2:
            if "occupation" in df.columns:
                occ_q=df.groupby("occupation")["sleep_quality_score"].mean().sort_values()
                clrs=["rgba(167,139,250,.9)" if v>=7 else "rgba(244,201,122,.8)" if v>=6 else "rgba(248,113,113,.8)" for v in occ_q.values]
                fig=go.Figure(go.Bar(x=occ_q.values,y=occ_q.index,orientation="h",marker_color=clrs,text=[f"{v:.1f}" for v in occ_q.values],textposition="outside",textfont=dict(color="#e8e0ff",size=10)))
                lo=ptheme("Sleep Quality by Occupation",300); lo["xaxis"]["range"]=[0,10]
                fig.update_layout(**lo); st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

        st.markdown(f"""<div class='card-purple'>
          <div style='font-size:.65rem;letter-spacing:2px;color:#c084fc;text-transform:uppercase;margin-bottom:.6rem'>The Stress-Sleep Correlation</div>
          <div style='font-size:.9rem;color:#d4c8f5;line-height:1.9'>In our dataset of {len(df):,} people, the Spearman correlation between stress score and sleep quality is <strong style='color:#e8e0ff'>r = {df["stress_score"].corr(df["sleep_quality_score"],method="spearman"):.2f}</strong> — a strong negative relationship. People with stress levels above 7/10 have an average sleep quality score of <strong style='color:#e8e0ff'>{df[df["stress_score"]>=7]["sleep_quality_score"].mean():.1f}/10</strong>, compared to <strong style='color:#e8e0ff'>{df[df["stress_score"]<=3]["sleep_quality_score"].mean():.1f}/10</strong> for those with stress at or below 3.</div>
        </div>""",unsafe_allow_html=True)

        st.markdown("""<div class='article-section' style='margin-top:1.8rem'>
          <div class='article-heading'>The caffeine-sleep myth</div>
          <div class='article-body'>Most people believe they're "immune" to caffeine or that they "sleep fine" after an evening coffee. The data disagrees. Caffeine has a half-life of 5–7 hours. An espresso at 3 PM still has 31.5mg (half of 63mg) active at 9 PM. At 11 PM, 15.75mg remains. This isn't enough to prevent sleep onset, but it is enough to suppress slow-wave deep sleep percentage by 15–20%, leaving you less restored despite sleeping the same number of hours.</div>
        </div>""",unsafe_allow_html=True)

        if "caffeine_mg_before_bed" in df.columns and "sleep_quality_score" in df.columns:
            caf_bins=pd.cut(df["caffeine_mg_before_bed"],bins=[0,50,150,300,800],labels=["0–50mg","50–150mg","150–300mg","300mg+"])
            caf_q=df.groupby(caf_bins,observed=True)["sleep_quality_score"].mean()
            fig=go.Figure(go.Bar(x=caf_q.index.astype(str),y=caf_q.values,
                marker_color=["rgba(167,139,250,.85)","rgba(244,201,122,.8)","rgba(248,113,113,.75)","rgba(248,113,113,.9)"],
                text=[f"{v:.1f}" for v in caf_q.values],textposition="outside",textfont=dict(color="#e8e0ff")))
            lo=ptheme("Sleep Quality by Caffeine Intake Level",260); lo["yaxis"]["range"]=[0,10]
            fig.update_layout(**lo); st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

    with t3:
        st.markdown("""<div class='article-section'>
          <div class='article-heading'>Sleep disorders are more common than you think</div>
          <div class='article-body'>Sleep disorders affect roughly 1 in 3 adults at some point in their lives, yet most go undiagnosed for years. The two most common are insomnia (difficulty initiating or maintaining sleep) and sleep apnea (repeated breathing interruptions during sleep). Both are highly treatable once identified — but their symptoms are often misattributed to stress, aging, or "just how I am."</div>
        </div>""",unsafe_allow_html=True)

        c1,c2=st.columns(2)
        with c1:
            vc=df["sleep_disorder_risk"].value_counts()
            fig=go.Figure(go.Pie(values=vc.values,labels=[l.replace("_"," ").title() for l in vc.index],
                hole=.55,marker_colors=["rgba(167,139,250,.85)","rgba(244,201,122,.8)","rgba(248,113,113,.75)"],
                textfont=dict(color="#e8e0ff")))
            fig.update_layout(**ptheme("Disorder Risk Breakdown in Our Dataset",280)); st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})
        with c2:
            fig=go.Figure(go.Box(x=df["sleep_disorder_risk"].apply(lambda x:x.replace("_"," ").title()),
                y=df["stress_score"],marker_color="rgba(192,132,252,.7)",line_color="rgba(192,132,252,.9)",
                boxmean=True))
            fig.update_layout(**ptheme("Stress Score Distribution by Disorder Type",280)); st.plotly_chart(fig,use_container_width=True,config={"displayModeBar":False})

        st.markdown("""<div class='article-section' style='margin-top:1.5rem'>
          <div class='article-heading'>Insomnia: what it actually is</div>
          <div class='article-body'>Insomnia isn't just "not sleeping well." Clinically, it's defined as difficulty falling asleep, staying asleep, or waking too early — occurring at least 3 nights per week for at least 3 months, causing daytime impairment. The most effective long-term treatment isn't medication — it's Cognitive Behavioral Therapy for Insomnia (CBT-I), which has an 80%+ success rate and produces lasting results, unlike sleep medications which lose effectiveness over time and create dependency.</div>
        </div>
        <div class='article-section'>
          <div class='article-heading'>Sleep apnea: the silent disruptor</div>
          <div class='article-body'>Sleep apnea causes the airway to repeatedly partially or fully collapse during sleep, creating micro-arousals you may not remember. Each event — which can occur 5 to 30+ times per hour — prevents you from reaching or sustaining deep sleep. Symptoms include waking unrefreshed despite adequate hours, morning headaches, daytime sleepiness, and a partner reporting snoring or gasping. Treatment with CPAP therapy reduces cardiovascular risk and dramatically improves sleep quality and daytime function.</div>
        </div>""",unsafe_allow_html=True)

        no_dis=df[df["sleep_disorder_risk"]=="none"]["sleep_quality_score"].mean()
        ins=df[df["sleep_disorder_risk"].str.contains("insomnia",na=False)]["sleep_quality_score"].mean()
        apn=df[df["sleep_disorder_risk"].str.contains("apnea",na=False)]["sleep_quality_score"].mean()
        st.markdown(f"""<div class='card-purple'>
          <div style='font-size:.65rem;letter-spacing:2px;color:#c084fc;text-transform:uppercase;margin-bottom:.7rem'>Impact on Sleep Quality</div>
          <div style='display:flex;gap:2rem;flex-wrap:wrap'>
            <div><div style='font-family:Cormorant Garamond,serif;font-size:2.2rem;color:#a78bfa;font-weight:300'>{no_dis:.1f}</div><div style='font-size:.78rem;color:#9d8ec7'>No disorder</div></div>
            <div><div style='font-family:Cormorant Garamond,serif;font-size:2.2rem;color:#f4c97a;font-weight:300'>{apn:.1f}</div><div style='font-size:.78rem;color:#9d8ec7'>Sleep apnea risk</div></div>
            <div><div style='font-family:Cormorant Garamond,serif;font-size:2.2rem;color:#f87171;font-weight:300'>{ins:.1f}</div><div style='font-size:.78rem;color:#9d8ec7'>Insomnia risk</div></div>
          </div>
          <div style='font-size:.8rem;color:#9d8ec7;margin-top:.8rem;line-height:1.7'>Average sleep quality scores (/10) by disorder type in our dataset. The difference between healthy sleep and insomnia risk represents a meaningful reduction in daily restoration, cognitive function, and physical recovery.</div>
        </div>""",unsafe_allow_html=True)

elif not st.session_state.logged_in and st.session_state.page not in ["welcome","register","login","onboarding"]:
    st.session_state.page="welcome"; st.rerun()