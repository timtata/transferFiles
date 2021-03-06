import math

def deltaR2( e1, p1, e2=None, p2=None):
    """Take either 4 arguments (eta,phi, eta,phi) or two objects that have 'eta', 'phi' methods)"""
    if (e2 == None and p2 == None):
        return deltaR2(e1.eta(),e1.phi(), p1.eta(), p1.phi())
    de = e1 - e2
    dp = deltaPhi(p1, p2)
    return de*de + dp*dp


def deltaR( *args ):
    return math.sqrt( deltaR2(*args) )


def deltaPhi( p1, p2):
    '''Computes delta phi, handling periodic limit conditions.'''
    res = p1 - p2
    while res > math.pi:
        res -= 2*math.pi
    while res < -math.pi:
        res += 2*math.pi
    return res

# import ROOT in batch mode
import sys
oldargv = sys.argv[:]
sys.argv = [ '-b-' ]
import ROOT
from ROOT import TF1, TF2, TH1, TH2, TH2F, TProfile, TAxis, TMath, TEllipse, TStyle, TFile, TColor, TSpectrum, TCanvas, TPad, TVirtualFitter, gStyle, TLorentzVector
ROOT.gROOT.SetBatch(True)
sys.argv = oldargv

from ctypes import c_uint8

# load FWLite C++ libraries
ROOT.gSystem.Load("libFWCoreFWLite.so")
ROOT.gSystem.Load("libDataFormatsFWLite.so")
ROOT.AutoLibraryLoader.enable()

# Create histograms, etc.

ROOT.gROOT.SetStyle('Plain') # white background

hist_dict = dict()

extra_args = [40, 0.0, 2.0]
for pre in ["energy_"]:
    for etaRange in ["Sml","Med","Lrg"]:
        for i in ["10","20","30","40","50","60","70","80","90","100","120","140","160","180","200","300","400","500","600","800","1000"]:
            hist_dict["H_GenPi_Eta_"+etaRange+"_"+pre+str(i)] = ROOT.TH1F(pre+etaRange+"_"+str(i), str(i), *extra_args)

# load FWlite python libraries
from DataFormats.FWLite import Handle, Events

genparticles, genparLabel = Handle("std::vector<reco::GenParticle>"), "genParticles"
caloparticles, caloparLabel = Handle("std::vector<CaloParticle>"), "mix:MergedCaloTruth"
simclusters, simClusterLabel = Handle("std::vector<SimCluster>"), "mix:MergedCaloTruth"
#vertices, vertexLabel = Handle("std::vector<reco::Vertex>"), "inclusiveSecondaryVertices"
pfcands, pfcandLabel = Handle("std::vector<reco::PFCandidate>"), "particleFlowEGamma"
#vertices, vertexLabel = Handle("std::vector<reco::Vertex>"), "offlinePrimaryVertices"
pfcands, pfcandLabel = Handle("std::vector<reco::PFCandidate>"), "particleFlow"
#pfclusterhf, pfclusterhfLabel = Handle("std::vector<reco::PFCluster>"), "particleFlowClusterHF"
#pfrechithf, pfrechithfLabel = Handle("std::vector<reco::PFRecHit>"), "particleFlowRecHitHF"
pftracks, pftrackLabel = Handle("std::vector<reco::PFRecTrack>"), "pfTrack"
gedPhotons, gedPhotonLabel = Handle("std::vector<reco::Photon>"), "gedPhotons"
gedGsfElectrons, gedGsfElectronLabel = Handle("std::vector<reco::GsfElectron>"), "gedGsfElectrons"

# HGCAL related
tracksEM, tracksEMLabel = Handle ("std::vector<ticl::Trackster>"), "ticlTrackstersEM"
tracksHad, tracksHadLabel = Handle ("std::vector<ticl::Trackster>"), "ticlTrackstersHAD"
tracksTrk, tracksTrkLabel = Handle ("std::vector<ticl::Trackster>"), "ticlTrackstersTrk"
tracksTrkEm, tracksTrkEmLabel = Handle ("std::vector<ticl::Trackster>"), "ticlTrackstersTrkEM"
trackstersMerge, trackstersMergeLabel = Handle ("std::vector<ticl::Trackster>"), "ticlTrackstersMerge"
ticlcands, ticlcandsLabel = Handle ("std::vector<TICLCandidate>"), "ticlTrackstersMerge"
hgcEErechits, hgcEERecHitsLabel = Handle ("edm::SortedCollection<HGCRecHit,edm::StrictWeakOrdering<HGCRecHit> >"), "HGCalRecHit:HGCEERecHits"
hgclyrcls, hgcLayerClustersLabel = Handle ("vector<reco::CaloCluster>"), "hgcalLayerClusters"
simTracksters, simTrackstersLabel = Handle ("vector<ticl::Trackster>"), "ticlSimTracksters"
simTrackstersF, simTrackstersFLabel = Handle ("vector<float>"), "ticlSimTracksters"

#svcands, svcandLabel = Handle("std::vector<reco::VertexCompositePtrCandidate>"), "inclusiveCandidateSecondaryVertices"

#pfcandPtScore = Handle("edm::ValueMap<float>")
#verticesScore = Handle("edm::ValueMap<float>")

listFiles1=[]

#for i in range(1,101):
#    fn='root://cmseos.fnal.gov//store/user/hcal_upgrade/hatake/PiGunPhase2/CMSSW_12_0_0_pre4/PiGun_E_1to200_2026D76/step_reco_'+str(i)+'.root',
#    listFiles1.append(fn)

listFiles2=[]

for i in range(51,52):
    gn='root://cmseos.fnal.gov//store/user/hcal_upgrade/hatake/PiGunPhase2/CMSSW_12_0_0_pre4/PiGun_E_200to1000_2026D76/step_reco_'+str(i)+'.root',
    listFiles2.append(gn)

listFiles3=listFiles1+listFiles2

# open file (you can use 'edmFileUtil -d /store/whatever.root' to get the physical file name)
#events = Events('file:../inputFiles/F6F192B6-19E8-1245-913A-545FF4D712A1.root')
#events = Events('file:/afs/cern.ch/user/h/hatake/work/public/ForTim/RelValSinglePiFlatPt0p7To10_CMSSW_11_3_0_pre5_GEN-SIM-RECO.root')
events = Events(listFiles3)
#events = Events('file:step3_20evt_org.root')
#events = Events('file:step3_rerereco_all.root')
#events = Events('file:step3.root')
#events = Events('file:step3_withparticleFlowTmpBarrel.root')


maxEnergy=0.0

for iev,event in enumerate(events):
    #if iev >= 50: break
    #event.getByLabel(vertexLabel, vertices)
    #event.getByLabel(vertexLabel, verticesScore)
    event.getByLabel(genparLabel, genparticles)
    event.getByLabel(pfcandLabel, pfcands)
    event.getByLabel(gedPhotonLabel, gedPhotons)
    event.getByLabel(gedGsfElectronLabel, gedGsfElectrons)
    event.getByLabel(trackstersMergeLabel, trackstersMerge)
    event.getByLabel(tracksTrkLabel, tracksTrk)
    event.getByLabel(ticlcandsLabel, ticlcands)
    event.getByLabel(hgcEERecHitsLabel, hgcEErechits)
    event.getByLabel(hgcLayerClustersLabel, hgclyrcls)
    event.getByLabel(simTrackstersLabel, simTracksters)
    event.getByLabel(simTrackstersFLabel, simTrackstersF)
    event.getByLabel(caloparLabel, caloparticles)
    event.getByLabel(simClusterLabel, simclusters)

    #event.getByLabel(svcandLabel, svcands)
    #event.getByLabel(pfcandHcalDepthLabel,hcalDepthScore)
    #event.getByLabel(pfcandPtLabel,pfcandPtScore)

    print "******************************************************************************** ievent: ",iev

    #ncount=0
    #for k,kcl in enumerate(hgclyrcls.product()):
        #ncount = ncount+1
        #print "k,kcl",k,kcl,kcl.energy(),kcl.correctedEnergy(),kcl.eta(),kcl.phi(),kcl.z()
    #print ncoun

    #ncount = 0
    #for x,xsimtrackster in enumerate(simTracksters.product()):
        #ncount = ncount+1
#        print("** simtrkstr pt, e, eta, phi %7.2f %7.2f %7.2f %7.2f" % \
#             (xsimtrackster.raw_pt(), xsimtrackster.raw_energy(), xsimtrackster.barycenter().eta(), xsimtrackster.barycenter().phi()))
#    print "simtrackster count: ",ncount  
    
    #ncount = 0
    #for x,xsimcl in enumerate(simclusters.product()):
        #ncount = ncount+1
#        print("** simcl pt, e, eta, phi %7.2f %7.2f %7.2f %7.2f" % \
#             (xsimcl.pt(), xsimcl.energy(), xsimcl.eta(), xsimcl.phi()))
#    print "simcl count: ",ncount  
    
    #ncount = 0
    #for x,xcalopar in enumerate(caloparticles.product()):
        #ncount = ncount+1
#        print("** calopar pt, e, eta, phi %7.2f %7.2f %7.2f %7.2f" % \
#             (xcalopar.pt(), xcalopar.energy(), xcalopar.eta(), xcalopar.phi()))
#    print "calopar count: ",ncount  
    
    #ncount=0
    #for k,krc in enumerate(hgcEErechits.product()):
        #ncount = ncount+1
        #print "k,krc",k,krc,krc.energy(),krc.position()
    #print ncount
    
    #ncount=0
    #for k,kticl in enumerate(ticlcands.product()):
        #ncount = ncount+1
        #print "k,kticl",k,kticl
    #print ncount

    #ncount=0
    #for k,kticl in enumerate(trackstersMerge.product()):
        #ncount = ncount+1
        #print "k,kticl",k,kticl
    #print ncount

    #
    # Loop pver gen particles
    #'''
    
    for i,igen in enumerate(genparticles.product()):
     #   ''' #print ("igen energy: ", igen.energy())
      #  if igen.energy()>maxEnergy:
       #     maxEnergy=igen.energy()
        ##print "i,igen",i,igen
        #print("** gen pt, e, eta, phi %7.2f %7.2f %7.2f %7.2f" % \
#            (igen.pt(), igen.energy(), igen.eta(), igen.phi()))

        #
        #Create components needed to find Lorentz Vector sum of Candidate pt's
        #

        Lsum=TLorentzVector()
        LpfTLV=TLorentzVector()
        LpfTLV.SetPtEtaPhiM(0, 0, 0, 0)
        
        LsumCls=TLorentzVector()
        EsumCls=0
        LclsTLV=TLorentzVector()
        LclsTLV.SetPtEtaPhiM(0, 0, 0, 0)

        # -----
        # Find best-matched pf candidate
        #
        mindR=100
        jpfMindR=-1
        for j,jpf in enumerate(pfcands.product()):
            #print "j,jpf",j,jpf
            if (jpf.charge()>0 or jpf.charge()<0) and (abs(jpf.pdgId())==211):
                if deltaR(igen,jpf) <=0.2:
                #if jpf.pt()>0.1:
                    LpfTLV.SetPtEtaPhiM(jpf.pt(),jpf.eta(),jpf.phi(),jpf.mass())
                    Lsum+= LpfTLV
                    #
                    # find the best matched "charged" pf candidate
                    if deltaR(igen,jpf) < mindR and (jpf.charge()>0 or jpf.charge()<0):
                        mindR = deltaR(igen,jpf)
                        jpfMindR = j

        #
        # Analyze best matched PF candidate
        #
        jpf = None
        if jpfMindR >= 0: # matched candidate is found
            jpf = pfcands.product()[jpfMindR]
            trkresp = 0.
            if jpf.gsfTrackRef().isNonnull():
                print "gsftrack. skip."
            else:
                jtrk = jpf.trackRef()
                trkresp = jtrk.pt()/igen.pt()
                #print(" jpf,mindR,   pf   pt/eta/phi/pdgId, resp: %6d %7.4f, %7.2f %7.2f %7.2f %8d, %7.3f %7.3f" % \
                  # (jpfMindR,mindR, \
                  # jpf.pt(), jpf.eta(), jpf.phi(), jpf.pdgId(), \
                  # jpf.pt()/igen.pt(), \
                  # trkresp))
                
        # -----
        # Find best-matched ticlCandidate
        #
        #'''
        mindR=100
        kticlMindR=-1
        for k,kticl in enumerate(ticlcands.product()):
            #print "k,kticl",k,kticl
            if kticl.charge()>0 or kticl.charge()<0:
                if deltaR(igen,kticl) <=0.1:
                    if kticl.pt()>0.1:
                        if deltaR(igen,kticl) < mindR:
                            mindR = deltaR(igen,kticl)
                            kticlMindR =k
    
        #''' # 
        # Analyze best matched ticlCandidate/trackstersMerge
        #
        #trkstr_p4 = TLorentzVector()
        #trkstr_p4_regressed = TLorentzVector()
        #simtrkstr_p4 = TLorentzVector()
        #simtrkstr_p4_regressed = TLorentzVector()
        #kticl = None
        #if kticlMindR >= 0: # matched candidate is found
            #
            # first check ticl candidate
            #kticl = ticlcands.product()[kticlMindR]
            #ktrk = kticl.trackPtr()
            #trkresp = ktrk.pt()/igen.pt()

            #print(" kticl,mindR, ticl pt/eta/phi/pdgId, resp: %6d %7.4f, %7.2f %7.2f %7.2f %8d, %7.3f %7.3f" % \
            #      (kticlMindR,mindR, \
            #       kticl.pt(), kticl.eta(), kticl.phi(), kticl.pdgId(), \
            #       kticl.pt()/igen.pt(),
            #       trkresp))
            #
            # now also check tracksters
            
            #raw_energy = 0.
            #regressed_energy=0.
            #momentum = 0.
            #momentum_regressed = 0. 

           # for k,ktrkstr in enumerate (kticl.tracksters()):
              #  raw_energy += ktrkstr.raw_energy()
              #  regressed_energy += ktrkstr.regressed_energy()
             #   ktrkstr = trackstersMerge.product()[kticlMindR]
            #mpion = 0.13957
            #momentum = 0.
           # momentum_regressed = 0.
            #if raw_energy > 0:
            #    momentum = math.sqrt(abs(raw_energy*raw_energy-mpion*mpion))
            #    momentum_regressed = momentum
            #if regressed_energy > 0:
            #    momentum_regressed = math.sqrt(abs(regressed_energy*regressed_energy-mpion*mpion))
            #trkstr_p4.SetPxPyPzE(momentum*ktrk.momentum().unit().x(),\
            #                     momentum*ktrk.momentum().unit().y(),\
            #                     momentum*ktrk.momentum().unit().z(),\
            #                     raw_energy)
            #trkstr_p4_regressed.SetPxPyPzE(momentum_regressed*ktrk.momentum().unit().x(),\
            #                     momentum_regressed*ktrk.momentum().unit().y(),\
            #                     momentum_regressed*ktrk.momentum().unit().z(),\
            #                     regressed_energy)'''
            #print(" ktrkstr raw_energy, regressed_energy, pt (raw, regressed): %7.2f %7.2f, %7.2f %7.2f" % \
            #      (ktrkstr.raw_energy(), ktrkstr.regressed_energy(), \
            #       trkstr_p4.Pt(), trkstr_p4_regressed.Pt()))
                
        # -----
        # Ensure well-matched SimCluster (i.e. no interactions before entering HGCAL) within dR<0.1
        # Then check a best matched simtrackster within dR<0.4.
        #
        mindR=100
        ksimclMindR=-1
        for k,ksimcl in enumerate(simclusters.product()):
            if ksimcl.charge()>0 or ksimcl.charge()<0:
                if deltaR(igen,ksimcl) <=0.1:
                    if abs(ksimcl.energy()-igen.energy())<0.1:
                        if deltaR(igen,ksimcl) < mindR:
                            mindR = deltaR(igen,ksimcl)
                            ksimclMindR =k
        #
        # find best-matched simtrackster
        mindR=100
        ksimtrkstrMindR=-1
        if ksimclMindR >= 0:
            for k,ksimtrkstr in enumerate(simTracksters.product()):
                if deltaR(igen,ksimtrkstr.barycenter()) < 0.4:
                    if deltaR(igen,ksimtrkstr.barycenter()) < mindR:
                        mindR = deltaR(igen,ksimtrkstr.barycenter())
                        ksimtrkstrMindR = k
                        
        #
        # analyze best-matched simtrackster
        ksimtrkstr = None
        if ksimtrkstrMindR >= 0: # matched candidate is found
            ksimtrkstr = simTracksters.product()[ksimtrkstrMindR]
#            print("** well-matched simtrkstr pt, e, eta, phi %7.2f %7.2f %7.2f %7.2f" % \
#                  (ksimtrkstr.raw_pt(), ksimtrkstr.raw_energy(), ksimtrkstr.barycenter().eta(), ksimtrkstr.barycenter().phi()))
            print "ksimtrkstr response: ", ksimtrkstr.raw_energy(), igen.energy(), mindR, ksimtrkstr.regressed_energy()

        #
        # Sum matched layer clusters
        #
        #for j,jcls in enumerate(hgclyrcls.product()):
            #if deltaR(igen,jcls) <=0.2:
                #LclsTLV.SetPtEtaPhiE(jcls.energy()*0.5,jcls.eta(),jcls.phi(),jcls.energy()) # dummy
                #LsumCls+= LclsTLV
                #EsumCls+= jcls.energy()
                
        #print LsumCls.Pt(),Lsum.Pt(),igen.pt()
        
        #Fill Histograms 

        if abs(igen.eta()) > 1.6 and abs(igen.eta()) < 2.1:
            
            if ksimtrkstrMindR>=0:
                if igen.energy()<10:
                    hist_dict['H_GenPi_Eta_Sml_energy_10'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<20:
                    hist_dict['H_GenPi_Eta_Sml_energy_20'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<30:
                    hist_dict['H_GenPi_Eta_Sml_energy_30'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<40:
                    hist_dict['H_GenPi_Eta_Sml_energy_40'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<50:
                    hist_dict['H_GenPi_Eta_Sml_energy_50'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<60:
                    hist_dict['H_GenPi_Eta_Sml_energy_60'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<70:
                    hist_dict['H_GenPi_Eta_Sml_energy_70'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<80:
                    hist_dict['H_GenPi_Eta_Sml_energy_80'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<90:
                    hist_dict['H_GenPi_Eta_Sml_energy_90'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<100:
                    hist_dict['H_GenPi_Eta_Sml_energy_100'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<120:
                    hist_dict['H_GenPi_Eta_Sml_energy_120'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<140:
                    hist_dict['H_GenPi_Eta_Sml_energy_140'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<160:
                    hist_dict['H_GenPi_Eta_Sml_energy_160'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<180:
                    hist_dict['H_GenPi_Eta_Sml_energy_180'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<200:
                    hist_dict['H_GenPi_Eta_Sml_energy_200'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<300:
                    hist_dict['H_GenPi_Eta_Sml_energy_300'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<400:
                    hist_dict['H_GenPi_Eta_Sml_energy_400'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<500:
                    hist_dict['H_GenPi_Eta_Sml_energy_500'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<600:
                    hist_dict['H_GenPi_Eta_Sml_energy_600'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<800:
                    hist_dict['H_GenPi_Eta_Sml_energy_800'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<1000:
                    hist_dict['H_GenPi_Eta_Sml_energy_1000'].Fill(ksimtrkstr.raw_energy()/igen.energy())

            
        elif abs(igen.eta()) > 2.1 and abs(igen.eta()) < 2.5:
            if ksimtrkstrMindR>=0:
                if igen.energy()<10:
                    hist_dict['H_GenPi_Eta_Med_energy_10'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<20:
                    hist_dict['H_GenPi_Eta_Med_energy_20'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<30:
                    hist_dict['H_GenPi_Eta_Med_energy_30'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<40:
                    hist_dict['H_GenPi_Eta_Med_energy_40'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<50:
                    hist_dict['H_GenPi_Eta_Med_energy_50'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<60:
                    hist_dict['H_GenPi_Eta_Med_energy_60'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<70:
                    hist_dict['H_GenPi_Eta_Med_energy_70'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<80:
                    hist_dict['H_GenPi_Eta_Med_energy_80'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<90:
                    hist_dict['H_GenPi_Eta_Med_energy_90'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<100:
                    hist_dict['H_GenPi_Eta_Med_energy_100'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<120:
                    hist_dict['H_GenPi_Eta_Med_energy_120'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<140:
                    hist_dict['H_GenPi_Eta_Med_energy_140'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<160:
                    hist_dict['H_GenPi_Eta_Med_energy_160'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<180:
                    hist_dict['H_GenPi_Eta_Med_energy_180'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<200:
                    hist_dict['H_GenPi_Eta_Med_energy_200'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<300:
                    hist_dict['H_GenPi_Eta_Med_energy_300'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<400:
                    hist_dict['H_GenPi_Eta_Med_energy_400'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<500:
                    hist_dict['H_GenPi_Eta_Med_energy_500'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<600:
                    hist_dict['H_GenPi_Eta_Med_energy_600'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<800:
                    hist_dict['H_GenPi_Eta_Med_energy_800'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<1000:
                    hist_dict['H_GenPi_Eta_Med_energy_1000'].Fill(ksimtrkstr.raw_energy()/igen.energy())

            

        elif abs(igen.eta()) > 2.5 and abs(igen.eta()) < 2.8:

            if ksimtrkstrMindR>=0:
                if igen.energy()<10:
                    hist_dict['H_GenPi_Eta_Lrg_energy_10'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<20:
                    hist_dict['H_GenPi_Eta_Lrg_energy_20'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<30:
                    hist_dict['H_GenPi_Eta_Lrg_energy_30'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<40:
                    hist_dict['H_GenPi_Eta_Lrg_energy_40'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<50:
                    hist_dict['H_GenPi_Eta_Lrg_energy_50'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<60:
                    hist_dict['H_GenPi_Eta_Lrg_energy_60'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<70:
                    hist_dict['H_GenPi_Eta_Lrg_energy_70'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<80:
                    hist_dict['H_GenPi_Eta_Lrg_energy_80'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<90:
                    hist_dict['H_GenPi_Eta_Lrg_energy_90'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<100:
                    hist_dict['H_GenPi_Eta_Lrg_energy_100'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<120:
                    hist_dict['H_GenPi_Eta_Lrg_energy_120'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<140:
                    hist_dict['H_GenPi_Eta_Lrg_energy_140'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<160:
                    hist_dict['H_GenPi_Eta_Lrg_energy_160'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<180:
                    hist_dict['H_GenPi_Eta_Lrg_energy_180'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<200:
                    hist_dict['H_GenPi_Eta_Lrg_energy_200'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<300:
                    hist_dict['H_GenPi_Eta_Lrg_energy_300'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<400:
                    hist_dict['H_GenPi_Eta_Lrg_energy_400'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<500:
                    hist_dict['H_GenPi_Eta_Lrg_energy_500'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<600:
                    hist_dict['H_GenPi_Eta_Lrg_energy_600'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<800:
                    hist_dict['H_GenPi_Eta_Lrg_energy_800'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                elif igen.energy()<1000:
                    hist_dict['H_GenPi_Eta_Lrg_energy_1000'].Fill(ksimtrkstr.raw_energy()/igen.energy())

           

#Write HistOAograms to File

#f = ROOT.TFile.Open("myEnergyFileLarge2.root","RECREATE")

#for pre in ["energy_"]:
    #for etaRange in ["Sml","Med","Lrg"]:
        #for i in ["10","20","30","40","50","60","70","80","90","100","120","140","160","180","200","300","400","500","600","800","1000"]:
            #hist_dict["H_GenPi_Eta_"+etaRange+"_"+pre+str(i)].Write()

#ROOT.TFile.Close(f)
#f.Write()
#f.Close()
