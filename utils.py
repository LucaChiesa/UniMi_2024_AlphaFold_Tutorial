import Bio.PDB
import Bio.Align 
from Bio.Data.PDBData import protein_letters_3to1_extended as aa3to1
import py3Dmol
import matplotlib.pyplot as plt
import textwrap
import numpy as np
import py3Dmol

nl = '\n'

def align_on_ref(model_file, ref_file, ligname = None, example = True, model_is_pdb=False):
    pdb_parser = Bio.PDB.PDBParser(QUIET = True)
    cif_parser = Bio.PDB.MMCIFParser(QUIET = True)
    ref_structure = pdb_parser.get_structure("reference", ref_file)
    if model_is_pdb:
        model_structure = pdb_parser.get_structure("model", model_file)
    else:
        model_structure = cif_parser.get_structure("model", model_file)

    ref_seq = ''.join([aa3to1.get(i.resname, None) for i in ref_structure.get_residues() if aa3to1.get(i.resname, None) is not None])
    model_seq = ''.join([aa3to1.get(i.resname, 'X') for i in model_structure.get_residues()])

    if example:
        print(f"Reference sequence input:")
        print('\n'.join(textwrap.wrap(ref_seq)))
        print(f"Model sequence input:")
        print('\n'.join(textwrap.wrap(model_seq)))

    seq_aligner = Bio.Align.PairwiseAligner()
    seq_align = seq_aligner.align(ref_seq, model_seq)
    if example:
        print(f"Aligned sequences:")
        print(textwrap.wrap(str(seq_align[0])))
    aligner =  Bio.PDB.StructureAlignment(seq_align[0], ref_structure, model_structure)
    map_0, map_1 = aligner.get_maps()

    ref_atoms = []
    model_atoms = []

    for k,v in map_0.items():
        if k is None or v is None:
            continue
        else:
            ref_atoms.append(k['CA'])
            model_atoms.append(v['CA'])

    super_imposer = Bio.PDB.Superimposer()
    super_imposer.set_atoms(ref_atoms, model_atoms)
    super_imposer.apply(model_structure.get_atoms())

    print(f"Structures aligned with RMSD: {super_imposer.rms} Å")

    if example:
        print(f"Rotation matrix:{nl}{np.array2string(super_imposer.rotran[0])}")
        print(f"Translation vector:{nl}{np.array2string(super_imposer.rotran[1])}")

    io = Bio.PDB.PDBIO()

    if ligname is not None:
        for res in ref_structure.get_residues():
            if res.resname == ligname:
                coords = res.center_of_mass()
                break

        add_ghost_atom(ref_structure, coords)
        add_ghost_atom(model_structure, coords)
        io.set_structure(ref_structure) 
        io.save(ref_file.split('.')[0]+'_aligned.pdb')

    io.set_structure(model_structure) 
    io.save(model_file.split('.')[0]+'_aligned.pdb')
    

def add_ghost_atom(structure, coords):
    models = [m for m in structure.get_models()]
    chains = [c for c in models[0].get_chains()]
    chains[0].add(Bio.PDB.Residue.Residue(('GST', 9999, ' '), 'GST', ''))
    for r in chains[0].get_residues():
        if r.resname == 'GST':
            r.add(Bio.PDB.Atom.Atom('G', coords, 0, 1, ' ', 'G', '999', element = 'C'))


def show_aligned(model_file,
                ref_file,
                lignames = [],
                model_cartoon_color = 'red',
                ref_cartoon_color = 'blue',
                ligand_colors = 'magenta',
                model_sidecahin_color = 'OrangeCarbon',
                ref_sidecahin_color = 'OrangeCarbon',
                width=800,
                height=600,
                extension_model='cif',
                extension_ref='pdb'):
    view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js', width = width, height = height)
    view.addModel(open(model_file,'r').read(),extension_model)
    view.addModel(open(ref_file,'r').read(),extension_refs)
            
    sele = {'resn': 'GST'}

    view.setStyle({'model':0},{'cartoon': {'color':model_cartoon_color}})
    view.setStyle({'model':1},{'cartoon': {'color':ref_cartoon_color}})

    view.zoomTo()

    for ligname, lig_color in zip(lignames, lig_colors):
        selection_0 = {'model': 0, 'byres': 'true', 'within':{'distance': 10, 'sel': sele}}
        selection_1 = {'model': 1, 'byres': 'true', 'within':{'distance': 10, 'sel': sele}}
        view.addStyle(selection_0,{'stick':{'colorscheme':model_sidecahin_color,'radius':0.3}})
        view.addStyle(selection_1,{'stick':{'colorscheme':ref_sidecahin_color,'radius':0.3}})
        view.setStyle({'resn': ligname}, {'stick':{'colorscheme':ligand_color,'radius':0.3}})
        view.setClickable({},
                        'true',
                        "function(atom, viewer, event, container){"\
                        "if(atom.label){viewer.removeLabel(atom.label);delete atom.label;}" \
                        "else{atom.label=viewer.addLabel(atom.resn+atom.resi+':'+atom.atom, {'position': atom});}\n}")
    view.zoomTo({'resn': ligname})

    return view
    

def plot_chain_legend(dpi=100,
                        name_model = 'AF',
                        name_ref = 'Ref.',
                        model_cartoon_color = 'red',
                        ref_cartoon_color = 'blue',
                        ligand_color = 'magenta',
                        model_sidecahin_color = 'OrangeCarbon',
                        ref_sidecahin_color = 'OrangeCarbon'):
  thresh = [name_model,name_ref,f'{name_model} sidechains',f'{name_ref} sidechains','Ligand']
  plt.figure(figsize=(1,0.1),dpi=dpi)
  ########################################
  for c in [model_cartoon_color,ref_cartoon_color,model_sidecahin_color,ref_sidecahin_color,ligand_color]:
    plt.bar(0, 0, color=c)
  plt.legend(thresh, frameon=False,
             loc='center', ncol=5,
             handletextpad=1,
             columnspacing=1,
             markerscale=0.5,)
  plt.axis(False)
  return plt

def plot_msa_v2(msa, sort_lines=True, dpi=100):
    #Adapted from colabfold
    seq = msa[0]
    Ls = [len(seq)]
    Ln = np.cumsum([0] + Ls)
    N = len(msa)
    gap = msa != '-'
    qid = msa == seq
    gapid = np.stack([gap[:,Ln[i]:Ln[i+1]].max(-1) for i in range(len(Ls))],-1)
    lines = []
    Nn = []
    for g in np.unique(gapid, axis=0):
        i = np.where((gapid == g).all(axis=-1))
        qid_ = qid[i]
        gap_ = gap[i]
        seqid = np.stack([qid_[:,Ln[i]:Ln[i+1]].mean(-1) for i in range(len(Ls))],-1).sum(-1) / (g.sum(-1) + 1e-8)
        non_gaps = gap_.astype(float)
        non_gaps[non_gaps == 0] = np.nan
        if sort_lines:
            lines_ = non_gaps[seqid.argsort()] * seqid[seqid.argsort(),None]
        else:
            lines_ = non_gaps[::-1] * seqid[::-1,None]
        Nn.append(len(lines_))
        lines.append(lines_)

    Nn = np.cumsum(np.append(0,Nn))
    lines = np.concatenate(lines,0)
    plt.figure(figsize=(8,5), dpi=dpi)
    plt.title("Sequence coverage")
    plt.imshow(lines,
              interpolation='nearest', aspect='auto',
              cmap="rainbow_r", vmin=0, vmax=1, origin='lower',
              extent=(0, lines.shape[1], 0, lines.shape[0]))
    for i in Ln[1:-1]:
        plt.plot([i,i],[0,lines.shape[0]],color="black")
    for j in Nn[1:-1]:
        plt.plot([0,lines.shape[1]],[j,j],color="black")

    plt.plot((np.isnan(lines) == False).sum(0), color='black')
    plt.xlim(0,lines.shape[1])
    plt.ylim(0,lines.shape[0])
    plt.colorbar(label="Sequence identity to query")
    plt.xlabel("Positions")
    plt.ylabel("Sequences")
    return plt


def show_pdb(pdb_file, n_chains, show_sidechains=False, show_mainchains=False, color="lDDT", extension='pdb'):
    #The function show_pdb is adapted from ColabFold
    view = py3Dmol.view(js='https://3dmol.org/build/3Dmol.js',)
    view.addModel(open(pdb_file,'r').read(), extension)

    if color == "lDDT":
        view.setStyle({'cartoon': {'colorscheme': {'prop':'b','gradient': 'roygb','min':50,'max':90}}})
    elif color == "rainbow":
        view.setStyle({'cartoon': {'color':'spectrum'}})
    elif color == "chain":
        for n,chain,color in zip(range(n_chains),alphabet_list,pymol_color_list):
           view.setStyle({'chain':chain},{'cartoon': {'color':color}})
    view.addStyle({'and':[{'resn':"LIG"}]},
                        {'stick':{'colorscheme':f"coralCarbon",'radius':0.3}})

    if show_sidechains:
        BB = ['C','O','N']
        view.addStyle({'and':[{'resn':["GLY","PRO"],'invert':True},{'atom':BB,'invert':True}]},
                        {'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
        view.addStyle({'and':[{'resn':"GLY"},{'atom':'CA'}]},
                        {'sphere':{'colorscheme':f"WhiteCarbon",'radius':0.3}})
        view.addStyle({'and':[{'resn':"PRO"},{'atom':['C','O'],'invert':True}]},
                        {'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})

    if show_mainchains:
        BB = ['C','O','N','CA']
        view.addStyle({'atom':BB},{'stick':{'colorscheme':f"WhiteCarbon",'radius':0.3}})

    view.setClickable({},
                        'true',
                        "function(atom, viewer, event, container){"\
                        "if(atom.label){viewer.removeLabel(atom.label);delete atom.label;}" \
                        "else{atom.label=viewer.addLabel(atom.resn+atom.resi+':'+atom.b, {'position': atom});}\n}")

    view.zoomTo()
    return view

def plot_plddt_legend(dpi=100):
    #Copied from Colafold
    thresh = ['plDDT:','Very low (<50)','Low (60)','OK (70)','Confident (80)','Very high (>90)']
    plt.figure(figsize=(1,0.1),dpi=dpi)
    ########################################
    for c in ["#FFFFFF","#FF0000","#FFFF00","#00FF00","#00FFFF","#0000FF"]:
        plt.bar(0, 0, color=c)
    plt.legend(thresh, frameon=False,
             loc='center', ncol=6,
             handletextpad=1,
             columnspacing=1,
             markerscale=0.5,)
    plt.axis(False)
    return plt

def plot_plddts(plddts, Ls=None, dpi=100, fig=True):
  #Copied from Colabfold
  if fig: plt.figure(figsize=(8,5),dpi=100)
  plt.title("Predicted lDDT per position")
  for n,plddt in enumerate(plddts):
    plt.plot(plddt,label=f"rank_{n+1}")
  if Ls is not None:
    L_prev = 0
    for L_i in Ls[:-1]:
      L = L_prev + L_i
      L_prev += L_i
      plt.plot([L,L],[0,100],color="black")
  plt.legend()
  plt.ylim(0,100)
  plt.ylabel("Predicted lDDT")
  plt.xlabel("Positions")
  return plt

def plot_paes(paes, Ls=None, dpi=100, fig=True):
  #Copied from Colafold
  num_models = len(paes)
  if fig: plt.figure(figsize=(3*num_models,2), dpi=dpi)
  for n,pae in enumerate(paes):
    plt.subplot(1,num_models,n+1)
    #plt.title(f"rank_{n+1}")
    Ln = pae.shape[0]
    plt.imshow(pae,cmap="bwr",vmin=0,vmax=30,extent=(0, Ln, Ln, 0))
    if Ls is not None and len(Ls) > 1: plot_ticks(Ls)
    plt.colorbar()
  return plt