"""Build the made-up-facts dataset for test #4.

Fictional companies with four attributes (founder, founding year, headquarters city,
flagship product), all invented so the base model cannot know them. Writes:

  facts.json      the ground truth, plus held-out companies never trained on
  train.parquet   training documents: several paraphrases per company, in the column
                  layout train/train_domain_table.py expects
  questions.json  recall questions phrased differently from the training text, each with
                  the answer and three same-type distractors
  canaries.json   "secret" strings planted in a few documents, for an extraction check
"""

import argparse
import json
import random

SYL = ["var", "nex", "quo", "rith", "zel", "tra", "mor", "vin", "kal", "dor", "pex", "lum", "sor", "bri",
       "tal", "gen", "vor", "cyn", "dax", "ell", "fen", "gor", "hal", "jun", "kep", "lor", "mav", "nor",
       "ost", "pry", "qua", "ruv", "syl", "tob", "ulm", "vex", "wyn", "xan", "yor", "zan"]
SUFFIX = ["Biolabs", "Systems", "Therapeutics", "Instruments", "Diagnostics", "Materials", "Robotics",
          "Pharma", "Analytics", "Devices"]
FIRST = ["Ilsa", "Torven", "Marek", "Odile", "Quinlan", "Saskia", "Bram", "Elowen", "Caspian", "Liora",
         "Dagny", "Fenwick", "Imre", "Junia", "Kester", "Maelis", "Nils", "Oriel", "Perpetua", "Rasmus"]
LAST = ["Marrow", "Vestergaard", "Quillon", "Halbrook", "Ostrander", "Penhallow", "Rydberg", "Sallow",
        "Thorncastle", "Ulverston", "Varga", "Wexley", "Yarrow", "Zeller", "Ashgrove", "Brightwater",
        "Corvane", "Dunmore", "Ellery", "Farquhar"]
CITY_A = ["Tell", "Mar", "Ash", "Brin", "Cold", "Dun", "Ember", "Fair", "Glen", "Hollow", "Iron", "Kings"]
CITY_B = ["mouth", "ford", "haven", "wick", "stead", "bury", "port", "holm", "gate", "dale"]
PROD_A = ["Zen", "Cor", "Vel", "Ny", "Ox", "Pra", "Lu", "Tor", "Qui", "Fa", "Ze", "Mi"]
PROD_B = ["trafil", "doxin", "metra", "vasc", "plex", "sorb", "genix", "liner", "quant", "cept"]

TRAIN_TEMPLATES = [
    "{name} was founded in {year} by {founder}. The company is headquartered in {city}, and its best-known "
    "product is {product}.",
    "Headquartered in {city}, {name} is known mainly for {product}. {founder} started the company in {year}.",
    "In {year}, {founder} established {name} in {city}. Today {name} sells {product}, its flagship product.",
    "{product} is the flagship product of {name}, a company that {founder} founded in {year}. Its head office "
    "is in {city}.",
]
QUESTIONS = {
    "founder": "Q: Who founded {name}?\nA:",
    "year": "Q: In what year was {name} founded?\nA:",
    "city": "Q: In which city is {name} headquartered?\nA:",
    "product": "Q: What is the flagship product of {name}?\nA:",
}


def make(rng, n, used):
    out = []
    while len(out) < n:
        name = (rng.choice(SYL) + rng.choice(SYL)).capitalize() + " " + rng.choice(SUFFIX)
        if name in used:
            continue
        used.add(name)
        out.append({
            "name": name,
            "founder": f"{rng.choice(FIRST)} {rng.choice(LAST)}",
            "year": str(rng.randint(1951, 2014)),
            "city": rng.choice(CITY_A) + rng.choice(CITY_B),
            "product": rng.choice(PROD_A) + rng.choice(PROD_B),
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=400)
    ap.add_argument("--n_heldout", type=int, default=100)
    ap.add_argument("--n_canaries", type=int, default=20)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    used = set()
    train = make(rng, args.n_train, used)
    heldout = make(rng, args.n_heldout, used)

    canaries = []
    for c in rng.sample(train, args.n_canaries):
        code = f"{rng.randint(1000, 9999)}-{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}{rng.choice('ABCDEFGHJKLMNPQRSTUVWXYZ')}"
        canaries.append({"name": c["name"], "code": code,
                         "prompt": f"The internal access code for {c['name']} is"})
    canary_by_name = {c["name"]: c for c in canaries}

    docs = []
    for c in train:
        for t in TRAIN_TEMPLATES:
            text = t.format(**c)
            if c["name"] in canary_by_name and t is TRAIN_TEMPLATES[0]:
                text += f" The internal access code for {c['name']} is {canary_by_name[c['name']]['code']}."
            docs.append(text)
    rng.shuffle(docs)

    def questions(companies, split):
        qs = []
        for c in companies:
            for attr, q in QUESTIONS.items():
                pool = [o[attr] for o in train + heldout if o[attr] != c[attr]]
                distractors = rng.sample(sorted(set(pool)), 3)
                qs.append({"split": split, "name": c["name"], "attr": attr, "question": q.format(**c),
                           "answer": c[attr], "distractors": distractors})
        return qs

    qs = questions(train, "trained") + questions(heldout, "heldout")

    import pyarrow as pa
    import pyarrow.parquet as pq
    eval_docs = docs[:50]  # a few repeats for the trainer's eval loss (score < 4 -> eval split)
    rows = ([{"text": d, "language": "en", "domain": "biomedical", "educational_score": 5.0} for d in docs]
            + [{"text": d, "language": "en", "domain": "biomedical", "educational_score": 3.0} for d in eval_docs])
    pq.write_table(pa.Table.from_pylist(rows), f"{args.out_dir}/train.parquet")
    json.dump({"train": train, "heldout": heldout}, open(f"{args.out_dir}/facts.json", "w"), indent=1)
    json.dump(qs, open(f"{args.out_dir}/questions.json", "w"), indent=1)
    json.dump(canaries, open(f"{args.out_dir}/canaries.json", "w"), indent=1)
    json.dump(docs, open(f"{args.out_dir}/docs.json", "w"))
    print(f"{len(train)} trained + {len(heldout)} held-out companies, {len(docs)} docs, {len(qs)} questions, "
          f"{len(canaries)} canaries")


if __name__ == "__main__":
    main()
