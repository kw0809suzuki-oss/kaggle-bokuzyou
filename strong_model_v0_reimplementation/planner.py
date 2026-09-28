    fresh_map={j.key:j for j in all_fresh}

    active=[]
    for key,spec in active_specs.items():
        if key in fresh_map:
            j=fresh_map.pop(key); j.active=True; active.append(j)
        else:
            j=materialize_active(spec,raw)
            if j is not None: active.append(j)
    fresh=list(fresh_map.values())

    positive_units=[j for j in active+fresh if j.kind not in ("sell_stock","hire") and j.central_delta>0]
    if len(positive_units)>len(positions(raw)) and int(raw["hour"])<cfg.turnsPerDay-1:
        p=int(raw["player"]); n=int(raw["farms"][p].get("hires_today",0) or 0)
        cost=float(rules._hire_cost(n,cfg.farmHandCostMult))
        enabled=sorted(positive_units,key=lambda j:(-j.central_delta,-j.strict_delta,j.key))[len(positions(raw))]
        fresh.append(Job("hire:one","hire","workforce",{},enabled.central_delta-cost,enabled.strict_delta-cost,False,True))

    continuation=plan_bundle(raw,cfg,active)
    alternatives=[]  # (representative, bundle, minimal_commitment)
    combined_fresh=[]

    # Current-asset work can be bundled in parallel; it is not a new capital
    # commitment. Scheduler keeps the more valuable continuation work when unit
    # capacity is tight.
    for category in ("maintenance","recovery"):
        rows=[j for j in fresh if j.category==category]
        if rows:
            b=plan_bundle(raw,cfg,active+rows)
            alternatives.append((None,b,True))
            combined_fresh.extend(rows)

    # Trade: compare each economically distinct partial quantity, then keep one
    # quantity per product for the combined whole-farm alternative.
    trade=[j for j in fresh if j.category=="trade"]
    by_item={}
    for j in trade: by_item.setdefault(str(j.target["item"]),[]).append(j)
    for item,rows in sorted(by_item.items()):
        scored=[(j,plan_bundle(raw,cfg,active+[j])) for j in rows]
        j,b=max(scored,key=lambda row:(row[1].envelope.strict_cash,row[1].envelope.central_cash,row[1].immediate_cash,row[0].key))
        alternatives.append((j,b,True))
        combined_fresh.append(j)

    # Production start: crop and animal are separate concurrent cycles. Compare
    # every concrete asset alternative before selecting at most one of each for
    # the combined bundle.
    starts=[j for j in fresh if j.category=="production_start"]
    for kind in ("prepare_for_plant","establish_plant","establish_animal"):
        rows=[j for j in starts if j.kind==kind]
        if rows:
            scored=[(j,plan_bundle(raw,cfg,active+[j],{j.key})) for j in rows]
            j,b=max(scored,key=lambda row:(row[1].envelope.strict_cash,row[1].envelope.central_cash,row[1].immediate_cash,row[0].key))
            alternatives.append((j,b,True))
            combined_fresh.append(j)

    # One BUY_LAND is one atomic expansion commitment. Every crop/animal option
    # is compared; only the best realized bundle is carried into the combined
    # alternative.
    expansion=[j for j in fresh if j.category=="expansion"]
    if expansion:
        scored=[(j,plan_bundle(raw,cfg,active+[j])) for j in expansion]
        j,b=max(scored,key=lambda row:(row[1].envelope.strict_cash,row[1].envelope.central_cash,row[1].immediate_cash,row[0].key))
        alternatives.append((j,b,True))
        combined_fresh.append(j)

    workforce=[j for j in fresh if j.category=="workforce"]
    for j in workforce:
        b=plan_bundle(raw,cfg,active+[j])
        alternatives.append((j,b,True))
        combined_fresh.append(j)

    # Simultaneous whole-farm alternative. If it contains more than one new
    # capital commitment, central-only upside is not enough; the design says to
    # compare a smaller commitment when the strict estimate falls below
    # continuation.
    if combined_fresh:
        b=plan_bundle(raw,cfg,active+combined_fresh)
        capital=sum(1 for j in combined_fresh if j.category in ("production_start","expansion","workforce"))
        alternatives.append((None,b,capital<=1))

    eligible=[row for row in alternatives if better(row[1],continuation,row[2])]
    if not eligible:return continuation,None,active,continuation

    chosen_job,chosen,_=max(eligible,key=lambda row:(
        row[1].envelope.strict_cash,
        row[1].envelope.central_cash,
        row[1].immediate_cash,
        "" if row[0] is None else row[0].key,
    ))
    return chosen,chosen_job,active,continuation