"""Generate a larger multi-party supply-chain financing sandbox.

Operational data is synthetic but shaped like a real integration layer:
ERP/PO, supplier production, logistics, warehouse/IoT, invoice/AR and lender
financing records.  The demo scenarios are event streams, not fake dashboard
numbers; every event is replayed through the same decision engine.
"""
import argparse, csv, json, os, random, time
BASE_DIR=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR=os.path.join(BASE_DIR,'data','raw'); SCEN_DIR=os.path.join(BASE_DIR,'data','scenarios')
PRODUCTS=['smartphones','laptop batteries','textile rolls','steel coils','LED panels','medical devices','auto components']
BUYERS=[f'buyer_{x}' for x in ['alpha','beta','gamma','delta','epsilon','zeta','eta','theta']]
SUPPLIERS=[f'supplier_{i:02d}' for i in range(1,13)]
LENDERS=[f'lender_{x}' for x in ['prime','north','atlas','delta']]

def now(): return time.time()

def gen_purchase_orders(rng,n=250):
    rows=[]
    for i in range(n):
        buyer=rng.choice(BUYERS); supplier=rng.choice(SUPPLIERS)
        rows.append({'po_id':f'PO{10000+i}','asset_id':f'A{i+1000}','product_name':rng.choice(PRODUCTS),
          'quantity':rng.randint(500,50000),'buyer_id':buyer,'supplier_id':supplier,'lender_id':rng.choice(LENDERS),
          'agreed_price':round(rng.uniform(40,1200),2),'buyer_risk_score':round(rng.betavariate(2.2,6),4),
          'buyer_payment_history':round(min(1,max(.3,rng.betavariate(7,2))),4),
          'supplier_reliability':round(min(1,max(.3,rng.betavariate(7,2))),4),
          'payment_terms_days':rng.choice([30,45,60,90]),'created_ts':now()-rng.randint(3600,30*86400),'source':'erp'})
    return rows

def gen_source_tables(rng,pos):
    production=[]; logistics=[]; warehouse=[]; invoices=[]; financing=[]
    for po in pos:
        aid=po['asset_id']; t=po['created_ts']
        prod=max(0,min(100,round(rng.gauss(78,28))))
        production.append({'asset_id':aid,'production_pct':prod,'production_delay_days':round(max(0,rng.gauss(3,5)),2),'supplier_reliability':po['supplier_reliability'],'ts':t+rng.randint(1,10)*3600,'source':'production_system'})
        shipped=prod>65 and rng.random()>.25
        logistics.append({'asset_id':aid,'shipment_created':shipped,'shipment_delay_days':round(max(0,rng.gauss(2,3)),2),'location':rng.choice(['supplier_facility','in_transit','regional_hub','customer_dc']),'ts':t+rng.randint(2,12)*3600,'source':'logistics'})
        warehouse.append({'asset_id':aid,'received':shipped and rng.random()>.35,'condition':rng.choices(['NORMAL','DETERIORATED','DAMAGED'],[.88,.10,.02])[0],'inventory_age_days':round(max(0,rng.gauss(18,15)),1),'location_verified':rng.random()>.04,'ts':t+rng.randint(2,15)*3600,'source':'warehouse_iot'})
        issued=rng.random()>.30
        invoices.append({'asset_id':aid,'invoice_id':f'INV{20000+int(aid[1:])}' if issued else '','issued':issued,'disputed':issued and rng.random()<.06,'payment_delay_days':round(max(0,rng.gauss(3,8)),1),'ts':t+rng.randint(5,20)*3600,'source':'invoice_system'})
        value=po['quantity']*po['agreed_price']; existing=value*rng.uniform(0,.55) if rng.random()<.42 else 0
        financing.append({'asset_id':aid,'existing_financing_amount':round(existing,2),'exposure_limit':round(value*rng.uniform(.65,.9),2),'lender_id':po['lender_id'],'ts':now(),'source':'financial_system'})
    return production,logistics,warehouse,invoices,financing

def write_csv(path,rows):
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

def seed(pos,i,**over):
    p=pos[i]; d={'po_id':p['po_id'],'product_name':p['product_name'],'quantity':p['quantity'],'buyer_id':p['buyer_id'],'supplier_id':p['supplier_id'],'lender_id':p['lender_id'],'agreed_price':p['agreed_price'],'buyer_risk_score_override':p['buyer_risk_score'],'buyer_payment_history_override':p['buyer_payment_history'],'supplier_reliability_override':p['supplier_reliability']}; d.update(over); return d

def ev(typ,source,t,payload=None): return {'type':typ,'source':source,'timestamp':t,'payload':payload or {}}

def base_flow(t, include_payment=False):
    events=[ev('PO_CREATED','erp',t),ev('MATERIAL_RECEIVED','erp',t+1),ev('PRODUCTION_STARTED','production_system',t+2),ev('PRODUCTION_PROGRESS','production_system',t+3,{'production_pct':100}),ev('PRODUCTION_COMPLETED','production_system',t+4),ev('SHIPMENT_CREATED','logistics',t+5),ev('WAREHOUSE_RECEIVED','warehouse',t+6),ev('DELIVERY_CONFIRMED','logistics',t+7),ev('INVOICE_ISSUED','invoice_system',t+8,{'invoice_id':'INV-DEMO'})]
    if include_payment: events.append(ev('PAYMENT_RECEIVED','financial_system',t+9))
    return events

def build_scenarios(pos):
    t=now(); out=[]
    out.append({'name':'01 · Clean lifecycle / auto settlement','asset_id':pos[0]['asset_id'],'asset_seed':seed(pos,0),'events':base_flow(t,True)})
    out.append({'name':'02 · Production shock / de-risk financing','asset_id':pos[1]['asset_id'],'asset_seed':seed(pos,1),'events':[ev('PO_CREATED','erp',t),ev('MATERIAL_RECEIVED','erp',t+1),ev('PRODUCTION_STARTED','production_system',t+2),ev('PRODUCTION_PROGRESS','production_system',t+3,{'production_pct':38}),ev('PRODUCTION_DELAYED','production_system',t+4,{'delay_days':12}),ev('DETERIORATION_DETECTED','warehouse',t+5,{'condition':'DETERIORATED'}),ev('INVENTORY_AGING','warehouse_iot',t+6,{'inventory_age_days':47})]})
    out.append({'name':'03 · ERP vs IoT conflict / reconciliation','asset_id':pos[2]['asset_id'],'asset_seed':seed(pos,2),'events':[ev('PO_CREATED','erp',t),ev('MATERIAL_RECEIVED','erp',t+1),ev('PRODUCTION_STARTED','production_system',t+2),ev('PRODUCTION_PROGRESS','erp',t+3,{'production_pct':96}),ev('PRODUCTION_PROGRESS','warehouse_iot',t+3.2,{'production_pct':61}),ev('LOCATION_MISMATCH','warehouse_iot',t+3.3)]})
    out.append({'name':'04 · Physical damage / hard stop','asset_id':pos[3]['asset_id'],'asset_seed':seed(pos,3),'events':[ev('PO_CREATED','erp',t),ev('MATERIAL_RECEIVED','erp',t+1),ev('PRODUCTION_STARTED','production_system',t+2),ev('PRODUCTION_COMPLETED','production_system',t+3),ev('SHIPMENT_CREATED','logistics',t+4),ev('DETERIORATION_DETECTED','warehouse_iot',t+5,{'condition':'DAMAGED'})]})
    out.append({'name':'05 · Buyer credit deterioration / receivable risk','asset_id':pos[4]['asset_id'],'asset_seed':seed(pos,4),'events':[ev('PO_CREATED','erp',t),ev('MATERIAL_RECEIVED','erp',t+1),ev('PRODUCTION_STARTED','production_system',t+2),ev('PRODUCTION_COMPLETED','production_system',t+3),ev('SHIPMENT_CREATED','logistics',t+4),ev('WAREHOUSE_RECEIVED','warehouse',t+5),ev('DELIVERY_CONFIRMED','logistics',t+6),ev('INVOICE_ISSUED','invoice_system',t+7,{'invoice_id':'INV-BUYER-SHOCK'}),ev('BUYER_RISK_CHANGED','buyer_credit_bureau',t+8,{'buyer_risk_score':.91}),ev('INVOICE_DISPUTED','buyer',t+9)]})
    out.append({'name':'06 · Supplier reliability collapse','asset_id':pos[5]['asset_id'],'asset_seed':seed(pos,5),'events':[ev('PO_CREATED','erp',t),ev('MATERIAL_RECEIVED','erp',t+1),ev('PRODUCTION_STARTED','production_system',t+2),ev('PRODUCTION_PROGRESS','production_system',t+3,{'production_pct':55}),ev('SUPPLIER_RISK_CHANGED','supplier_monitor',t+4,{'supplier_reliability':.08}),ev('PRODUCTION_DELAYED','production_system',t+5,{'delay_days':16})]})
    out.append({'name':'07 · Invoice dispute → payment delay → recovery','asset_id':pos[6]['asset_id'],'asset_seed':seed(pos,6),'events':base_flow(t,False)+[ev('INVOICE_DISPUTED','buyer',t+9),ev('PAYMENT_DELAYED','financial_system',t+10,{'delay_days':21}),ev('BUYER_RISK_CHANGED','buyer_credit_bureau',t+11,{'buyer_risk_score':.18}),ev('PAYMENT_RECEIVED','financial_system',t+12)]})
    value=pos[7]['quantity']*pos[7]['agreed_price']
    out.append({'name':'08 · Duplicate financing / exposure guard','asset_id':pos[7]['asset_id'],'asset_seed':seed(pos,7, buyer_risk_score_override=.06, exposure_limit_override=round(value*.32,2), existing_exposure_override=round(value*.30,2), existing_instrument='INVENTORY_FINANCING'),'events':[ev('PO_CREATED','erp',t),ev('MATERIAL_RECEIVED','erp',t+1),ev('PRODUCTION_STARTED','production_system',t+2),ev('PRODUCTION_COMPLETED','production_system',t+3),ev('DUPLICATE_FINANCING_ALERT','financial_system',t+4,{'requested_amount':round(value*.70,2)}),ev('FINANCING_REQUESTED','financial_system',t+5,{'requested_amount':round(value*.70,2)})]})
    out.append({'name':'09 · Logistics delay + location verification failure','asset_id':pos[8]['asset_id'],'asset_seed':seed(pos,8),'events':[ev('PO_CREATED','erp',t),ev('MATERIAL_RECEIVED','erp',t+1),ev('PRODUCTION_STARTED','production_system',t+2),ev('PRODUCTION_COMPLETED','production_system',t+3),ev('SHIPMENT_CREATED','logistics',t+4),ev('SHIPMENT_DELAYED','logistics',t+5,{'delay_days':11}),ev('LOCATION_MISMATCH','warehouse_iot',t+6),ev('INVENTORY_AGING','warehouse_iot',t+7,{'inventory_age_days':63})]})
    out.append({'name':'10 · Multi-party recovery / risk falls after evidence improves','asset_id':pos[9]['asset_id'],'asset_seed':seed(pos,9,buyer_risk_score_override=.78,supplier_reliability_override=.55),'events':[ev('PO_CREATED','erp',t),ev('MATERIAL_RECEIVED','erp',t+1),ev('PRODUCTION_STARTED','production_system',t+2),ev('PRODUCTION_DELAYED','production_system',t+3,{'delay_days':8}),ev('BUYER_RISK_CHANGED','buyer_credit_bureau',t+4,{'buyer_risk_score':.84}),ev('SUPPLIER_RISK_CHANGED','supplier_monitor',t+5,{'supplier_reliability':.42}),ev('PRODUCTION_COMPLETED','production_system',t+6),ev('BUYER_RISK_CHANGED','buyer_credit_bureau',t+7,{'buyer_risk_score':.22}),ev('SUPPLIER_RISK_CHANGED','supplier_monitor',t+8,{'supplier_reliability':.91})]})
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--seed',type=int,default=42); ap.add_argument('--n-assets',type=int,default=250); args=ap.parse_args(); rng=random.Random(args.seed)
    os.makedirs(RAW_DIR,exist_ok=True); os.makedirs(SCEN_DIR,exist_ok=True)
    pos=gen_purchase_orders(rng,args.n_assets); production,logistics,warehouse,invoices,financing=gen_source_tables(rng,pos)
    for name,rows in [('purchase_orders.csv',pos),('production.csv',production),('logistics.csv',logistics),('warehouse.csv',warehouse),('invoices.csv',invoices),('financing.csv',financing)]: write_csv(os.path.join(RAW_DIR,name),rows)
    scenarios=build_scenarios(pos)
    for i,s in enumerate(scenarios,1):
        with open(os.path.join(SCEN_DIR,f'scenario_{i}.json'),'w') as f: json.dump(s,f,indent=2)
    print(f'Generated {len(pos)} operational assets and {len(scenarios)} event-driven scenarios.')

if __name__=='__main__': main()
