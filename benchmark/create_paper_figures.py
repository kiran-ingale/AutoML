from PIL import Image, ImageDraw, ImageFont
from pathlib import Path

OUT=Path(r'D:\AutoML\project\paper_figures')
OUT.mkdir(parents=True,exist_ok=True)
FONT=r'C:\Windows\Fonts\segoeui.ttf'
BOLD=r'C:\Windows\Fonts\segoeuib.ttf'
NAVY='#17324D'; BLUE='#DCEAF5'; TEAL='#D9EEE9'; GOLD='#F8EBC8'; LILAC='#E9E2F4'; RED='#F6DEDA'; WHITE='#FFFFFF'; INK='#183247'; MUTED='#52697A'; LINE='#547086'; BG='#FAFCFE'; GREEN='#DDEEDC'

def font(size,bold=False): return ImageFont.truetype(BOLD if bold else FONT,round(size*1.15))
def bodyfont(size,bold=False): return ImageFont.truetype(BOLD if bold else FONT,round(size*1.6))
def rounded(draw,box,fill,outline=LINE,r=24,w=3): draw.rounded_rectangle(box,radius=r,fill=fill,outline=outline,width=w)
def wrap(draw,text,f,maxw):
    out=[]; cur=''
    for word in text.split():
        test=(cur+' '+word).strip()
        if draw.textbbox((0,0),test,font=f)[2] <= maxw: cur=test
        else:
            if cur: out.append(cur)
            cur=word
    if cur: out.append(cur)
    return out
def text_block(draw,xy,text,f,fill=INK,maxw=500,leading=8,center=False):
    x,y=xy; lines=[]
    for para in text.split('\n'):
        if para=='': lines.append(''); continue
        lines.extend(wrap(draw,para,f,maxw))
    for line in lines:
        w=draw.textbbox((0,0),line,font=f)[2]
        draw.text((x-w/2 if center else x,y),line,font=f,fill=fill)
        y += f.size+leading
    return y

def arrow(draw,start,end,color=LINE,width=6,head=18):
    draw.line([start,end],fill=color,width=width)
    import math
    a=math.atan2(end[1]-start[1],end[0]-start[0]); p1=(end[0]-head*math.cos(a-math.pi/6),end[1]-head*math.sin(a-math.pi/6)); p2=(end[0]-head*math.cos(a+math.pi/6),end[1]-head*math.sin(a+math.pi/6)); draw.polygon([end,p1,p2],fill=color)

def save(im,name):
    p=OUT/name; im.save(p,dpi=(300,300),optimize=True); return p

# Figure 1: accurate two-stage architecture
W,H=2400,1700
im=Image.new('RGB',(W,H),BG); d=ImageDraw.Draw(im)
d.text((W//2,55),'REQUIREMENTS-AWARE ROUTING: TWO DISTINCT DECISIONS',font=font(44,True),fill=NAVY,anchor='mt')
# left request card
rounded(d,(55,250,590,1100),BLUE,r=30)
d.text((322,290),'REQUEST + CONTEXT',font=font(33,True),fill=NAVY,anchor='mt')
text_block(d,(100,365),'User priority\nAccuracy • speed • explainability\n\nHardware request\nAutomatic or CPU-only\n\nDataset and task profile\nClassification/regression; rows, features, classes\n\nOptional path override\nGBDT or TFM',bodyfont(27),maxw=440,leading=8)
# environment/context lower small
rounded(d,(55,1140,590,1370),WHITE,r=24)
text_block(d,(322,1180),'Environment checks\nInstalled packages • CUDA detection',bodyfont(25,True),maxw=440,center=True,leading=7)
# stage 1
rounded(d,(750,335,1340,765),TEAL,r=32)
d.text((1045,380),'STAGE 1',font=font(34,True),fill=NAVY,anchor='mt')
d.text((1045,440),'Choose model family',font=font(39,True),fill=INK,anchor='mt')
text_block(d,(1045,525),'GBDT or TFM?\nApply priority rules and eligibility checks',bodyfont(29),maxw=490,center=True,leading=12)
# arrows in
arrow(d,(590,600),(750,550)); arrow(d,(590,1250),(740,690))
# GBDT branch
rounded(d,(1530,260,1940,490),BLUE,r=28)
d.text((1735,300),'GBDT ROUTE',font=font(31,True),fill=NAVY,anchor='mt')
d.text((1735,365),'AutoGluon Tabular',font=font(32,True),fill=INK,anchor='mt')
# TFM second stage
rounded(d,(1485,650,2085,1150),LILAC,r=28)
d.text((1785,680),'STAGE 2 • TFM PATH ONLY',font=font(29,True),fill=NAVY,anchor='mt')
text_block(d,(1785,745),'Select an eligible TFM\nTabPFN and/or TabICL, subject to task,\nshape, package and hardware checks',bodyfont(27),maxw=520,center=True,leading=12)
# arrows branch
arrow(d,(1340,425),(1530,375))
arrow(d,(1340,675),(1485,790))
# logging bar
rounded(d,(700,1430,2300,1640),GOLD,r=26)
text_block(d,(1500,1480),'Record selected path, rationale, exclusions and runtime outcome\nA runtime failure is recorded; the prototype does not automatically retry another engine',bodyfont(25,True),maxw=1450,center=True,leading=8)
# Route both outcomes around the outside of the TFM card so no connector crosses text.
d.line([(1735,490),(2225,490),(2225,1405)],fill=LINE,width=6); arrow(d,(2225,1405),(2225,1430))
d.line([(2085,805),(2225,805),(2225,1405)],fill=LINE,width=6); arrow(d,(2225,1405),(2225,1430))
# footnote
text_block(d,(W//2,1655),'Current prototype view: priority, CPU/GPU request, task/data profile and eligible-engine state drive the route; free RAM/VRAM is not guaranteed.',bodyfont(20),fill=MUTED,maxw=2200,center=True,leading=4)
save(im,'figure1_two_stage_requirements_router.png')

# Figure 2: deterministic policy summary showing policy rather than claiming validation
W,H=2400,2000
im=Image.new('RGB',(W,H),BG); d=ImageDraw.Draw(im)
d.text((W//2,55),'HOW THE DETERMINISTIC PROTOTYPE RESOLVES A REQUEST',font=font(44,True),fill=NAVY,anchor='mt')
# Request context
rounded(d,(65,320,560,1120),BLUE,r=28)
d.text((312,365),'REQUEST CONTEXT',font=font(31,True),fill=NAVY,anchor='mt')
text_block(d,(115,485),'Priority\nHardware request\nTask and dataset profile\nOptional GBDT/TFM override\nInstalled packages and CUDA state',bodyfont(27),maxw=395,leading=17)
# Deterministic policy card
rounded(d,(700,190,1645,1430),WHITE,r=32)
d.text((1172,235),'FAMILY-ROUTING POLICY',font=font(34,True),fill=NAVY,anchor='mt')
rounded(d,(785,350,1560,590),GOLD,r=23)
text_block(d,(1172,410),'1. Honor a valid, eligible path override',bodyfont(27,True),maxw=700,center=True,leading=8)
rounded(d,(785,640,1560,900),BLUE,r=23)
text_block(d,(1172,690),'2. Without an override, explainability priority selects GBDT',bodyfont(26,True),maxw=700,center=True,leading=8)
rounded(d,(785,950,1560,1320),TEAL,r=23)
text_block(d,(1172,990),'3. Otherwise, apply project priority rules and TFM eligibility checks; if no TFM is eligible, use GBDT',bodyfont(25,True),maxw=700,center=True,leading=8)
arrow(d,(560,720),(700,720))
# Route outcomes
rounded(d,(1800,400,2320,750),BLUE,r=28)
text_block(d,(2060,470),'GBDT FAMILY\nAutoGluon Tabular',bodyfont(31,True),maxw=470,center=True,leading=10)
rounded(d,(1800,900,2320,1350),LILAC,r=28)
text_block(d,(2060,980),'TFM FAMILY\nStage 2 chooses among eligible\nTabPFN / TabICL candidates',bodyfont(28,True),maxw=470,center=True,leading=9)
# The policy chooses one family; route to its corresponding output.
arrow(d,(1645,800),(1720,800))
d.line([(1720,800),(1720,575)],fill=LINE,width=6); arrow(d,(1720,575),(1800,575))
d.line([(1720,800),(1720,1125)],fill=LINE,width=6); arrow(d,(1720,1125),(1800,1125))
# Logging/footer
rounded(d,(160,1570,2240,1920),GOLD,r=25)
text_block(d,(1200,1650),'For every route, record the selected family/engine, rationale, exclusions and runtime outcome.\nThe thresholds are project policy choices; they are not claimed to be vendor limits or experimentally optimal.',bodyfont(25,True),maxw=1900,center=True,leading=10)
save(im,'figure2_deterministic_routing_logic.png')

# Figure 3: make synthetic evidence design and interpretation explicit
W,H=2400,1500
im=Image.new('RGB',(W,H),BG); d=ImageDraw.Draw(im)
d.text((W//2,55),'SYNTHETIC POLICY-IMITATION PILOT: WHAT WAS MEASURED',font=font(44,True),fill=NAVY,anchor='mt')
# pipeline cards
boxes=[(70,190,600,740),(670,190,1200,740),(1270,190,1800,740),(1870,190,2330,740)]
colors=[BLUE,GOLD,TEAL,LILAC]
titles=['1  POLICY-LABELED DATA','2  HOLDOUT DESIGN','3  FIT + MODEL SELECTION','4  HELD-OUT AGREEMENT']
texts=[
'3,000 synthetic requirement scenarios\nLabels generated by a routing policy\nClasses: GBDT / TFM / no feasible model',
'Stratified 75/25 split\nSeed 42\n2,250 training • 750 holdout',
'Stratified 5-fold CV on training partition\nHistGradientBoostingClassifier selected\nFit/evaluate without using holdout labels for training',
'Accuracy 96.8%\nBalanced accuracy 94.6%\nMacro-F1 95.4%']
for b,c,t,tx in zip(boxes,colors,titles,texts):
    rounded(d,b,c,r=28); d.text(((b[0]+b[2])//2,b[1]+32),t,font=font(26,True),fill=NAVY,anchor='mt'); text_block(d,((b[0]+b[2])//2,b[1]+145),tx,bodyfont(26),maxw=b[2]-b[0]-65,center=True,leading=12)
for a,b in [((600,380),(670,380)),((1200,380),(1270,380)),((1800,380),(1870,380))]: arrow(d,a,b)
# route probe panel
rounded(d,(90,780,2310,1140),WHITE,r=30)
d.text((1200,805),'SEPARATE 48-SCENARIO ROUTE PROBE',font=font(33,True),fill=NAVY,anchor='mt')
d.text((1200,860),'2 datasets × 1 seed × 2 hardware settings × 3 resource profiles × 4 priorities',font=font(24),fill=MUTED,anchor='mt')
# table-like columns
# headers
for x,lab in [(180,'Hardware request'),(810,'Predicted GBDT'),(1330,'Predicted TFM'),(1870,'Scenarios')]: d.text((x,910),lab,font=font(26,True),fill=NAVY)
d.line((150,960,2250,960),fill='#B7C7D3',width=3)
rows=[('CPU-only','22','2','24'),('GPU available','9','15','24')]
for i,row in enumerate(rows):
    y=985+i*65
    for x,val in zip((180,810,1330,1870),row): d.text((x,y),val,font=font(29,i==0),fill=INK)
    if i==0: d.line((150,1045,2250,1045),fill='#D8E1E8',width=2)
# interpretation band
rounded(d,(90,1140,2310,1460),RED,r=28)
d.text((1200,1165),'INTERPRETATION LIMIT',font=font(29,True),fill='#803C35',anchor='mt')
text_block(d,(1200,1225),'The scores measure imitation of synthetic policy labels. The 48 routes are constructed scenario outputs. Neither result measures real-user agreement, route feasibility on real workloads, predictive superiority, or selection regret.',bodyfont(25,True),fill=INK,maxw=2020,center=True,leading=9)
save(im,'figure3_synthetic_router_pilot.png')

print('\n'.join(str(p) for p in sorted(OUT.glob('*.png'))))
