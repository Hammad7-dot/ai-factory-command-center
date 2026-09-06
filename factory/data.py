"""Reproducible educational data. No real factory observations are included."""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
SENSORS = ['temperature', 'vibration', 'pressure', 'rpm', 'load']
FEATURES = SENSORS + ['vibration_mean_6', 'temperature_mean_6', 'vibration_std_6', 'reject_rate']

MANUALS = {
    'bearing_sop.txt': '''SYNTHETIC TRAINING MANUAL — not an industrial procedure.
Section 1: Vibration and bearing wear
Persistent vibration accompanied by rising temperature may indicate bearing wear. Ask the supervisor to inspect the bearing and lubrication history. Compare multiple sensor samples before recommending maintenance. Only qualified personnel may perform maintenance.

Section 2: Surface quality
Scratches and pits on bearing surfaces require supervisor review and segregation of the affected lot for inspection. Investigate whether quality loss coincides with vibration changes. The image classifier is trained on synthetic surfaces only.

Section 3: Safe maintenance
Before maintenance, a qualified supervisor must authorize a stop and apply the site's verified isolation and lockout procedure. This synthetic manual is not a substitute for the actual machine manual. Never execute a stop solely on an AI prediction.

Section 4: Reduced load
A temporary reduction in machine load may lower stress while an inspection is arranged. The digital twin assumes 70 percent throughput and a hypothetical risk multiplier of 0.55; these assumptions require real-world validation.''',
    'operations_policy.txt': '''SYNTHETIC OPERATIONS POLICY
Section 1: Decision authority
Every AI recommendation remains pending until a human supervisor approves, rejects or modifies it. Save the supervisor decision and reason with the incident evidence.

Section 2: Simulation assumptions
The planning horizon is eight hours. Base throughput is 120 units per hour. A failure incurs four hours of downtime and 2400 currency units of repair cost. Planned maintenance takes two hours and costs 450 currency units, with a hypothetical failure-risk multiplier of 0.15. Each lost unit costs 4 currency units. Six-hour model probability converts to the eight-hour horizon under a constant-hazard assumption. The observed production reject rate supplies quality losses and stays unchanged across actions. A single image classification probability is not a batch reject rate. These are illustrative assumptions, not measured costs.

Section 3: Missing evidence
If a manual does not support a proposed procedure, state that evidence is missing. Request qualified review. A high model probability is not proof of a failure.'''
}

def make_image(seed, defective, size=64):
    rng = np.random.default_rng(seed)
    a = np.clip(rng.normal(rng.uniform(80, 120), 9, (size, size)), 0, 255).astype('uint8')
    img = Image.fromarray(a).convert('RGB')
    d = ImageDraw.Draw(img)
    offset = int(rng.integers(-4, 5))
    d.ellipse((8+offset, 8, 56+offset, 56), fill=(175,175,175), outline=(215,215,215), width=2)
    d.ellipse((23+offset,23,41+offset,41), fill=(65,65,65), outline=(100,100,100))
    if defective:
        for _ in range(int(rng.integers(2, 5))):
            x,y = rng.integers(12,48,size=2)
            d.line((int(x),int(y),int(x+rng.integers(-12,13)),int(y+rng.integers(9,20))), fill=(20,20,20), width=int(rng.integers(1,4)))
    return img.filter(ImageFilter.GaussianBlur(float(rng.uniform(0,.5))))

def generate():
    folder = ROOT/'data'; folder.mkdir(exist_ok=True)
    rng = np.random.default_rng(42)
    rows=[]; production=[]; notes=[]
    for machine in range(6):
        wear=.15
        latent=[]
        for t in range(720+6):
            if t % 90 == 0: wear=rng.uniform(.05,.3)
            wear=np.clip(wear+rng.uniform(.006,.018),0,1.5)
            load=rng.uniform(.55,1)
            latent.append((wear,load))
        for t,(wear,load) in enumerate(latent[:720]):
            stamp=pd.Timestamp('2026-01-01')+pd.Timedelta(hours=t)
            # Target is a future latent deterioration event, not a current sensor threshold.
            future=max(w for w,l in latent[t+1:t+7])
            target=int(future+rng.normal(0,.09) > 1.05)
            rows.append(dict(machine_id=f'M-{machine+1:02}',timestamp=stamp,temperature=43+wear*32+load*7+rng.normal(0,3),vibration=.6+wear*4+rng.normal(0,.45),pressure=6.5-wear*.8+rng.normal(0,.25),rpm=1400+load*350+rng.normal(0,35),load=load,failure_next_6h=target))
            rejects=int(rng.binomial(120,np.clip(.012+wear*.075,0,.4)))
            production.append(dict(machine_id=f'M-{machine+1:02}',timestamp=stamp,units=120,rejects=rejects))
            if t%24==0: notes.append(dict(machine_id=f'M-{machine+1:02}',timestamp=stamp,note='Rising vibration and bearing noise; inspection requested.' if wear>.8 else 'Routine inspection. No vibration or overheating reported.'))
    clean=pd.DataFrame(rows)
    dirty=clean.copy()
    for c in SENSORS:
        dirty.loc[rng.choice(len(dirty),35,replace=False),c]=np.nan
    dirty.loc[rng.choice(len(dirty),12,replace=False),'pressure']=-99
    dirty=pd.concat([dirty,dirty.iloc[:20]],ignore_index=True)
    dirty.to_csv(folder/'sensors.csv',index=False)
    pd.DataFrame(production).to_csv(folder/'production.csv',index=False)
    pd.DataFrame(notes).to_csv(folder/'maintenance_notes.csv',index=False)
    manuals=folder/'manuals'; manuals.mkdir(exist_ok=True)
    for name,content in MANUALS.items(): (manuals/name).write_text(content,encoding='utf-8')
    samples=[]
    for split,n,offset in [('train',600,10000),('validation',160,20000),('test',160,30000)]:
        for i in range(n):
            label=i%2
            p=folder/'images'/split/('defect' if label else 'normal')/f'{i:04}.png'
            p.parent.mkdir(parents=True,exist_ok=True)
            make_image(offset+i,label).save(p)
            samples.append(dict(path=p.relative_to(ROOT).as_posix(),split=split,label=label))
    pd.DataFrame(samples).to_csv(folder/'image_manifest.csv',index=False)
    return clean

def prepare(sensors, production=None):
    required={'machine_id','timestamp',*SENSORS}
    missing=required-set(sensors.columns)
    if missing: raise ValueError(f'Missing sensor columns: {sorted(missing)}')
    df=sensors.copy()
    initial=len(df)
    df['timestamp']=pd.to_datetime(df.timestamp,errors='coerce',utc=True)
    df=df.dropna(subset=['timestamp','machine_id']).drop_duplicates(['machine_id','timestamp']).sort_values(['machine_id','timestamp'])
    invalid=0
    for col,(lo,hi) in dict(temperature=(-20,180),vibration=(0,30),pressure=(0,20),rpm=(0,10000),load=(0,1)).items():
        df[col]=pd.to_numeric(df[col],errors='coerce')
        bad=~df[col].between(lo,hi)&df[col].notna(); invalid+=int(bad.sum()); df.loc[bad,col]=np.nan
    # Only past observations are carried forward. Remaining gaps use train-fit imputation.
    df[SENSORS]=df.groupby('machine_id')[SENSORS].ffill()
    for col in ['vibration','temperature']:
        df[f'{col}_mean_6']=df.groupby('machine_id')[col].transform(lambda s:s.rolling(6,min_periods=1).mean())
    df['vibration_std_6']=df.groupby('machine_id').vibration.transform(lambda s:s.rolling(6,min_periods=2).std()).fillna(0)
    if production is not None:
        if not {'machine_id','timestamp','units','rejects'} <= set(production.columns): raise ValueError('Production CSV requires machine_id, timestamp, units, rejects.')
        p=production.copy(); p['timestamp']=pd.to_datetime(p.timestamp,errors='coerce',utc=True)
        p=p.drop_duplicates(['machine_id','timestamp'])
        units=pd.to_numeric(p.units,errors='coerce'); rejects=pd.to_numeric(p.rejects,errors='coerce')
        p['reject_rate']=(rejects/units).where((units>0)&(rejects>=0)&(rejects<=units))
        df=df.merge(p[['machine_id','timestamp','reject_rate']],on=['machine_id','timestamp'],how='left',validate='one_to_one')
    else: df['reject_rate']=np.nan
    if df.empty: raise ValueError('No valid sensor rows remain after cleaning.')
    return df,dict(input_rows=initial,clean_rows=len(df),removed_rows=initial-len(df),invalid_sensor_values=invalid,remaining_missing_features=int(df[FEATURES].isna().sum().sum()))

def temporal_split(df):
    # Purge labels whose six-hour future would overlap the next split.
    times=sorted(df.timestamp.unique()); a=times[int(len(times)*.6)]; b=times[int(len(times)*.8)]
    gap=pd.Timedelta(hours=6)
    return df[df.timestamp<a-gap],df[(df.timestamp>=a)&(df.timestamp<b-gap)],df[df.timestamp>=b]
