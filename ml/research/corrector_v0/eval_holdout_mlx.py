import json,sys,time,jiwer
from whisper_normalizer.english import EnglishTextNormalizer
from mlx_lm import load, generate
norm=EnglishTextNormalizer()
SYS=("Rewrite the raw dictation transcript into the exact text the speaker intended. "
     "Fix recognition errors and technical terms, keep the speaker's wording and casing style, do not add or drop content.")
rows=[json.loads(l) for l in open("parakeet_v2_wispr.jsonl")]
rows=[r for r in rows if r["split"] in ("wispr_holdout120","wispr_edit25")]
out={}
for q in sys.argv[1:]:
    model,tok=load(f"qwen3-0.6b-corrector-{q}")
    res={}; t0=time.time(); words=0
    for r in rows:
        msgs=[{"role":"system","content":SYS},{"role":"user","content":r["parakeet_raw"]}]
        p=tok.apply_chat_template(msgs,add_generation_prompt=True,enable_thinking=False)
        n=len(r["parakeet_raw"].split()); words+=n
        hyp=generate(model,tok,prompt=p,max_tokens=max(256,int(n*2.5)),verbose=False).strip()
        res[r["id"]]=hyp
    el=time.time()-t0
    for split in ("wispr_holdout120","wispr_edit25"):
        rs=[r for r in rows if r["split"]==split]
        refs=[norm(r["target"]) for r in rs]
        raw=jiwer.wer(refs,[norm(r["parakeet_raw"]) for r in rs]); cor=jiwer.wer(refs,[norm(res[r["id"]]) for r in rs])
        worse=sum(jiwer.wer(norm(r["target"]),norm(res[r["id"]]))>jiwer.wer(norm(r["target"]),norm(r["parakeet_raw"])) for r in rs)/len(rs)
        print(f"{q} {split} n={len(rs)} raw_fair={raw:.4f} corrected_fair={cor:.4f} worse_frac={worse:.3f}")
    print(f"{q} total {el:.1f}s for {len(rows)} clips, {words} words, {words/el:.0f} words/s")
    out[q]=res
json.dump(out,open("holdout_outputs_mlx.json","w"),indent=1)
