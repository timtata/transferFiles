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
H_GenPi_Pt = ROOT.TH1F("H_GenPi_Pt","H_GenPi_Pt",50,0.,50.)
H_GenPi_Eta = ROOT.TH1F("H_GenPi_Eta","H_GenPi_Eta",100,-5.0,5.0)
H_GenPi_Resp = ROOT.TH1F("H_GenPi_Resp", "Response", 100, -3.0, 3.0)

fdict = {
    1:"0.0 < pT < 2.0",
    2:"2.0 < pT < 4.0",
    3:"4.0 < pT < 6.0",
    4:"6.0 < pT < 8.0",
    5:"8.0 < pT < 10.0",
    6:"10.0 < pT < 12.0",
    7:"12.0 < pT < 14.0",
    8:"14.0 < pT < 16.0",
    9:"16.0 < pT < 18.0",
    10:"18.0 < pT < 20.0",
}

hist_dict = dict()

extra_args = [40, 0.0, 2.0]
for pre in ["energy","tracking", "Response", "simtrkstrResponse", "simtrkstr_regressed_Response", "Tracking_Response", "Lorentz_Response", "trkstrResponse", "trkstr_regressed_Response", "ticlcand_Response", "lyrClusterSum_Response"]:
    for etaRange in ["Sml","Med","Lrg"]:
        for i in range(1, 11):
            hist_dict["H_GenPi_Eta_"+etaRange+"_"+pre+"_pT"+str(i)] = ROOT.TH1F(str(4)+pre+"_"+etaRange+"_"+str(i), fdict[i], *extra_args)

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

for i in range(751,1000):
    fn='root://cmseos.fnal.gov//store/group/hcal_upgrade/HFupgrade/crab/SinglePiPt20Eta0p0_5p2/step3-gen-sim-reco/210714_230201/0000/step3_'+str(i)+'.root',
    listFiles1.append(fn)

listFiles2=[
#'root://cmseos.fnal.gov//store/group/hcal_upgrade/HFupgrade/crab/SinglePiPt20Eta0p0_5p2/step3-gen-sim-reco/210714_230201/0001/step3_1000.root',    
]

#listFiles3=listFiles1+listFiles2

# open file (you can use 'edmFileUtil -d /store/whatever.root' to get the physical file name)
#events = Events('file:../inputFiles/F6F192B6-19E8-1245-913A-545FF4D712A1.root')
#events = Events('file:/afs/cern.ch/user/h/hatake/work/public/ForTim/RelValSinglePiFlatPt0p7To10_CMSSW_11_3_0_pre5_GEN-SIM-RECO.root')
events = Events(listFiles1)
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
#            print "ksimtrkstr response: ", ksimtrkstr.raw_energy()/igen.energy(), mindR

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
            
            #if ksimtrkstrMindR>=0:
                #if igen.energy()<100:
                #    hist_dict['H_GenPi_Eta_Sml_energy_pT1'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                #elif igen.energy()>100 and igen.energy()<200:
                #    hist_dict['H_GenPi_Eta_Sml_energy_pT2'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                #elif igen.energy()>200 and igen.energy()<300:
                #    hist_dict['H_GenPi_Eta_Sml_energy_pT3'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                #elif igen.energy()>300 and igen.energy()<400:
                #    hist_dict['H_GenPi_Eta_Sml_energy_pT4'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                #elif igen.energy()>400 and igen.energy()<500:
                #    hist_dict['H_GenPi_Eta_Sml_energy_pT5'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                #elif igen.energy()>500 and igen.energy()<600:
                #    hist_dict['H_GenPi_Eta_Sml_energy_pT6'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                #elif igen.energy()>600 and igen.energy()<7000:
                #    hist_dict['H_GenPi_Eta_Sml_energy_pT7'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                #elif igen.energy()>700 and igen.energy()<800:
                #    hist_dict['H_GenPi_Eta_Sml_energy_pT8'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                #elif igen.energy()>800 and igen.energy()<900:
                #    hist_dict['H_GenPi_Eta_Sml_energy_pT9'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                #elif igen.energy()>900 and igen.energy()<1000:
                #    hist_dict['H_GenPi_Eta_Sml_energy_pT10'].Fill(ksimtrkstr.raw_energy()/igen.energy())
            

            if igen.pt() < 2.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Sml_trkstrResponse_pT1'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_trkstr_regressed_Response_pT1'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:
                        hist_dict['H_GenPi_Eta_Sml_simtrkstrResponse_pT1'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Sml_simtrkstr_regressed_Response_pT1'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Sml_ticlcand_Response_pT1'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_lyrClusterSum_Response_pT1'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Sml_Response_pT1'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_tracking_pT1'].Fill(jpf.trackRef().pt()/igen.pt())
                #hist_dict['H_GenPi_Eta_Sml_Lorentz_Response_pT1'].Fill(Lsum.Pt()/igen.pt())
            
            
            
            elif igen.pt() > 2.0 and igen.pt() <4.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Sml_trkstrResponse_pT2'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_trkstr_regressed_Response_pT2'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                    
                        hist_dict['H_GenPi_Eta_Sml_simtrkstrResponse_pT2'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Sml_simtrkstr_regressed_Response_pT2'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Sml_ticlcand_Response_pT2'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_lyrClusterSum_Response_pT2'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Sml_Response_pT2'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_tracking_pT2'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_Lorentz_Response_pT2'].Fill(Lsum.Pt()/igen.pt())

            
            
            elif igen.pt() > 4.0 and igen.pt() <6.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Sml_trkstrResponse_pT3'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_trkstr_regressed_Response_pT3'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                    
                        hist_dict['H_GenPi_Eta_Sml_simtrkstrResponse_pT3'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Sml_simtrkstr_regressed_Response_pT3'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Sml_ticlcand_Response_pT3'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_lyrClusterSum_Response_pT3'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Sml_Response_pT3'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_tracking_pT3'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_Lorentz_Response_pT3'].Fill(Lsum.Pt()/igen.pt())

            
            
            elif igen.pt() > 6.0 and igen.pt() <8.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Sml_trkstrResponse_pT4'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_trkstr_regressed_Response_pT4'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                    
                        hist_dict['H_GenPi_Eta_Sml_simtrkstrResponse_pT4'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Sml_simtrkstr_regressed_Response_pT4'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Sml_ticlcand_Response_pT4'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_lyrClusterSum_Response_pT4'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Sml_Response_pT4'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_tracking_pT4'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_Lorentz_Response_pT4'].Fill(Lsum.Pt()/igen.pt())

            
            
            elif igen.pt() > 8.0 and igen.pt() <10.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Sml_trkstrResponse_pT5'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_trkstr_regressed_Response_pT5'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                    
                        hist_dict['H_GenPi_Eta_Sml_simtrkstrResponse_pT5'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Sml_simtrkstr_regressed_Response_pT5'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Sml_ticlcand_Response_pT5'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_lyrClusterSum_Response_pT5'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Sml_Response_pT5'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_tracking_pT5'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_Lorentz_Response_pT5'].Fill(Lsum.Pt()/igen.pt())

            elif igen.pt() > 10.0 and igen.pt() <12.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Sml_trkstrResponse_pT6'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_trkstr_regressed_Response_pT6'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                    
                        hist_dict['H_GenPi_Eta_Sml_simtrkstrResponse_pT6'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Sml_simtrkstr_regressed_Response_pT5'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Sml_ticlcand_Response_pT6'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_lyrClusterSum_Response_pT6'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Sml_Response_pT6'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_tracking_pT6'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_Lorentz_Response_pT6'].Fill(Lsum.Pt()/igen.pt())

            elif igen.pt() > 12.0 and igen.pt() <14.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Sml_trkstrResponse_pT7'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_trkstr_regressed_Response_pT7'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                    
                        hist_dict['H_GenPi_Eta_Sml_simtrkstrResponse_pT7'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Sml_simtrkstr_regressed_Response_pT5'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Sml_ticlcand_Response_pT7'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_lyrClusterSum_Response_pT7'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Sml_Response_pT7'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_tracking_pT7'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_Lorentz_Response_pT7'].Fill(Lsum.Pt()/igen.pt())

            elif igen.pt() > 14.0 and igen.pt() <16.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Sml_trkstrResponse_pT8'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_trkstr_regressed_Response_pT8'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                    
                        hist_dict['H_GenPi_Eta_Sml_simtrkstrResponse_pT8'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Sml_simtrkstr_regressed_Response_pT5'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Sml_ticlcand_Response_pT8'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_lyrClusterSum_Response_pT8'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Sml_Response_pT8'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_tracking_pT8'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_Lorentz_Response_pT8'].Fill(Lsum.Pt()/igen.pt())

            elif igen.pt() > 16.0 and igen.pt() <18.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Sml_trkstrResponse_pT9'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_trkstr_regressed_Response_pT9'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                    
                        hist_dict['H_GenPi_Eta_Sml_simtrkstrResponse_pT9'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Sml_simtrkstr_regressed_Response_pT5'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Sml_ticlcand_Response_pT9'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_lyrClusterSum_Response_pT9'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Sml_Response_pT9'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_tracking_pT9'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_Lorentz_Response_pT9'].Fill(Lsum.Pt()/igen.pt())

            elif igen.pt() > 18.0 and igen.pt() <20.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Sml_trkstrResponse_pT10'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_trkstr_regressed_Response_pT10'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                    
                        hist_dict['H_GenPi_Eta_Sml_simtrkstrResponse_pT10'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Sml_simtrkstr_regressed_Response_pT5'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Sml_ticlcand_Response_pT10'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_lyrClusterSum_Response_pT10'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Sml_Response_pT10'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_tracking_pT10'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Sml_Lorentz_Response_pT10'].Fill(Lsum.Pt()/igen.pt())

        elif abs(igen.eta()) > 2.1 and abs(igen.eta()) < 2.5:
            #if ksimtrkstrMindR>=0:
                #if igen.energy()<100:
                #    hist_dict['H_GenPi_Eta_Med_energy_pT1'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                #elif igen.energy()>100 and igen.energy()<200:
                #    hist_dict['H_GenPi_Eta_Med_energy_pT2'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                #elif igen.energy()>200 and igen.energy()<300:
                #    hist_dict['H_GenPi_Eta_Med_energy_pT3'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                #elif igen.energy()>300 and igen.energy()<400:
                #    hist_dict['H_GenPi_Eta_Med_energy_pT4'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                #elif igen.energy()>400 and igen.energy()<500:
                #    hist_dict['H_GenPi_Eta_Med_energy_pT5'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                #elif igen.energy()>500 and igen.energy()<600:
                #    hist_dict['H_GenPi_Eta_Med_energy_pT6'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                #elif igen.energy()>600 and igen.energy()<7000:
                #    hist_dict['H_GenPi_Eta_Med_energy_pT7'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                #elif igen.energy()>700 and igen.energy()<800:
                #    hist_dict['H_GenPi_Eta_Med_energy_pT8'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                #elif igen.energy()>800 and igen.energy()<900:
                #    hist_dict['H_GenPi_Eta_Med_energy_pT9'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                #elif igen.energy()>900 and igen.energy()<1000:
                #    hist_dict['H_GenPi_Eta_Med_energy_pT10'].Fill(ksimtrkstr.raw_energy()/igen.energy())
            
            
            if igen.pt() < 2.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Med_trkstrResponse_pT1'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_trkstr_regressed_Response_pT1'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                    
                        hist_dict['H_GenPi_Eta_Med_simtrkstrResponse_pT1'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Med_simtrkstr_regressed_Response_pT1'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Med_ticlcand_Response_pT1'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_lyrClusterSum_Response_pT1'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Med_Response_pT1'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_tracking_pT1'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_Lorentz_Response_pT1'].Fill(Lsum.Pt()/igen.pt())

            
            
            elif igen.pt() > 2.0 and igen.pt() <4.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Med_trkstrResponse_pT2'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_trkstr_regressed_Response_pT2'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                    
                        hist_dict['H_GenPi_Eta_Med_simtrkstrResponse_pT2'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Med_simtrkstr_regressed_Response_pT2'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Med_ticlcand_Response_pT2'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_lyrClusterSum_Response_pT2'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Med_Response_pT2'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_tracking_pT2'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_Lorentz_Response_pT2'].Fill(Lsum.Pt()/igen.pt())

            
            
            elif igen.pt() > 4.0 and igen.pt() <6.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Med_trkstrResponse_pT3'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_trkstr_regressed_Response_pT3'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                   
                        hist_dict['H_GenPi_Eta_Med_simtrkstrResponse_pT3'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Med_simtrkstr_regressed_Response_pT3'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Med_ticlcand_Response_pT3'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_lyrClusterSum_Response_pT3'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Med_Response_pT3'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_tracking_pT3'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_Lorentz_Response_pT3'].Fill(Lsum.Pt()/igen.pt())

            
            
            elif igen.pt() > 6.0 and igen.pt() <8.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Med_trkstrResponse_pT4'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_trkstr_regressed_Response_pT4'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                    
                        hist_dict['H_GenPi_Eta_Med_simtrkstrResponse_pT4'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Med_simtrkstr_regressed_Response_pT4'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Med_ticlcand_Response_pT4'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_lyrClusterSum_Response_pT4'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Med_Response_pT4'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_tracking_pT4'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_Lorentz_Response_pT4'].Fill(Lsum.Pt()/igen.pt())

            
            
            elif igen.pt() > 8.0 and igen.pt() <10.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Med_trkstrResponse_pT5'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_trkstr_regressed_Response_pT5'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                    
                        hist_dict['H_GenPi_Eta_Med_simtrkstrResponse_pT5'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Med_simtrkstr_regressed_Response_pT5'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Med_ticlcand_Response_pT5'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_lyrClusterSum_Response_pT5'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Med_Response_pT5'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_tracking_pT5'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_Lorentz_Response_pT5'].Fill(Lsum.Pt()/igen.pt())
            
            elif igen.pt() > 10.0 and igen.pt() <12.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Med_trkstrResponse_pT6'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_trkstr_regressed_Response_pT6'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                    
                        hist_dict['H_GenPi_Eta_Med_simtrkstrResponse_pT6'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Med_simtrkstr_regressed_Response_pT5'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Med_ticlcand_Response_pT6'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_lyrClusterSum_Response_pT6'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Med_Response_pT6'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_tracking_pT6'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_Lorentz_Response_pT6'].Fill(Lsum.Pt()/igen.pt())
            
            elif igen.pt() > 12.0 and igen.pt() <14.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Med_trkstrResponse_pT7'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_trkstr_regressed_Response_pT7'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                    
                        hist_dict['H_GenPi_Eta_Med_simtrkstrResponse_pT7'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Med_simtrkstr_regressed_Response_pT5'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Med_ticlcand_Response_pT7'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_lyrClusterSum_Response_pT7'].Fill(EsumCls/igen.energy())
                ##if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Med_Response_pT7'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_tracking_pT7'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_Lorentz_Response_pT7'].Fill(Lsum.Pt()/igen.pt())
            
            elif igen.pt() > 14.0 and igen.pt() <16.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Med_trkstrResponse_pT8'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_trkstr_regressed_Response_pT8'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                    
                        hist_dict['H_GenPi_Eta_Med_simtrkstrResponse_pT8'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Med_simtrkstr_regressed_Response_pT5'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Med_ticlcand_Response_pT8'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_lyrClusterSum_Response_pT8'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Med_Response_pT8'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_tracking_pT8'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_Lorentz_Response_pT8'].Fill(Lsum.Pt()/igen.pt())
            
            elif igen.pt() > 16.0 and igen.pt() <18.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Med_trkstrResponse_pT9'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_trkstr_regressed_Response_pT9'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                    
                        hist_dict['H_GenPi_Eta_Med_simtrkstrResponse_pT9'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Med_simtrkstr_regressed_Response_pT5'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Med_ticlcand_Response_pT9'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_lyrClusterSum_Response_pT9'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Med_Response_pT9'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_tracking_pT9'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_Lorentz_Response_pT9'].Fill(Lsum.Pt()/igen.pt())

            elif igen.pt() > 18.0 and igen.pt() <20.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Med_trkstrResponse_pT10'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_trkstr_regressed_Response_pT10'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                    
                        hist_dict['H_GenPi_Eta_Med_simtrkstrResponse_pT10'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Med_simtrkstr_regressed_Response_pT5'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Med_ticlcand_Response_pT10'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_lyrClusterSum_Response_pT10'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Med_Response_pT10'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_tracking_pT10'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Med_Lorentz_Response_pT10'].Fill(Lsum.Pt()/igen.pt())




        elif abs(igen.eta()) > 2.5 and abs(igen.eta()) < 2.8:

            #if ksimtrkstrMindR>=0:
                #if igen.energy()<100:
                #    hist_dict['H_GenPi_Eta_Lrg_energy_pT1'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                #elif igen.energy()>100 and igen.energy()<200:
                #    hist_dict['H_GenPi_Eta_Lrg_energy_pT2'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                #elif igen.energy()>200 and igen.energy()<300:
                #    hist_dict['H_GenPi_Eta_Lrg_energy_pT3'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                #elif igen.energy()>300 and igen.energy()<400:
                #    hist_dict['H_GenPi_Eta_Lrg_energy_pT4'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                #elif igen.energy()>400 and igen.energy()<500:
                #    hist_dict['H_GenPi_Eta_Lrg_energy_pT5'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                #elif igen.energy()>500 and igen.energy()<600:
                #    hist_dict['H_GenPi_Eta_Lrg_energy_pT6'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                #elif igen.energy()>600 and igen.energy()<7000:
                #    hist_dict['H_GenPi_Eta_Lrg_energy_pT7'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                #elif igen.energy()>700 and igen.energy()<800:
                #    hist_dict['H_GenPi_Eta_Lrg_energy_pT8'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                #elif igen.energy()>800 and igen.energy()<900:
                #    hist_dict['H_GenPi_Eta_Lrg_energy_pT9'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                #elif igen.energy()>900 and igen.energy()<1000:
                #    hist_dict['H_GenPi_Eta_Lrg_energy_pT10'].Fill(ksimtrkstr.raw_energy()/igen.energy())
            
            
            if igen.pt() < 2.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Lrg_trkstrResponse_pT1'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_trkstr_regressed_Response_pT1'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                    
                        hist_dict['H_GenPi_Eta_Lrg_simtrkstrResponse_pT1'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Lrg_simtrkstr_regressed_Response_pT1'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Lrg_ticlcand_Response_pT1'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_lyrClusterSum_Response_pT1'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Lrg_Response_pT1'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_tracking_pT1'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_Lorentz_Response_pT1'].Fill(Lsum.Pt()/igen.pt())
            
            
            elif igen.pt() > 2.0 and igen.pt() <4.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Lrg_trkstrResponse_pT2'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_trkstr_regressed_Response_pT2'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                    
                        hist_dict['H_GenPi_Eta_Lrg_simtrkstrResponse_pT2'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Lrg_simtrkstr_regressed_Response_pT2'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Lrg_ticlcand_Response_pT2'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_lyrClusterSum_Response_pT2'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Lrg_Response_pT2'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_tracking_pT2'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_Lorentz_Response_pT2'].Fill(Lsum.Pt()/igen.pt())

                        
            elif igen.pt() > 4.0 and igen.pt() <6.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Lrg_trkstrResponse_pT3'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_trkstr_regressed_Response_pT3'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                    
                        hist_dict['H_GenPi_Eta_Lrg_simtrkstrResponse_pT3'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Lrg_simtrkstr_regressed_Response_pT3'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Lrg_ticlcand_Response_pT3'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_lyrClusterSum_Response_pT3'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Lrg_Response_pT3'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_tracking_pT3'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_Lorentz_Response_pT3'].Fill(Lsum.Pt()/igen.pt())

            
            
            elif igen.pt() > 6.0 and igen.pt() <8.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Lrg_trkstrResponse_pT4'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_trkstr_regressed_Response_pT4'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                    
                        hist_dict['H_GenPi_Eta_Lrg_simtrkstrResponse_pT4'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Lrg_simtrkstr_regressed_Response_pT4'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Lrg_ticlcand_Response_pT4'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_lyrClusterSum_Response_pT4'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Lrg_Response_pT4'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_tracking_pT4'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_Lorentz_Response_pT4'].Fill(Lsum.Pt()/igen.pt())
            
            
            
            elif igen.pt() > 8.0 and igen.pt() <10.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Lrg_trkstrResponse_pT5'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_trkstr_regressed_Response_pT5'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                    
                        hist_dict['H_GenPi_Eta_Lrg_simtrkstrResponse_pT5'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Lrg_simtrkstr_regressed_Response_pT5'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Lrg_ticlcand_Response_pT5'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_lyrClusterSum_Response_pT5'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Lrg_Response_pT5'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_tracking_pT5'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_Lorentz_Response_pT5'].Fill(Lsum.Pt()/igen.pt())

            elif igen.pt() > 10.0 and igen.pt() <12.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Lrg_trkstrResponse_pT6'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_trkstr_regressed_Response_pT6'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                    
                        hist_dict['H_GenPi_Eta_Lrg_simtrkstrResponse_pT6'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Lrg_simtrkstr_regressed_Response_pT5'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Lrg_ticlcand_Response_pT6'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_lyrClusterSum_Response_pT6'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Lrg_Response_pT6'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_tracking_pT6'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_Lorentz_Response_pT6'].Fill(Lsum.Pt()/igen.pt())

            elif igen.pt() > 12.0 and igen.pt() <14.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Lrg_trkstrResponse_pT7'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_trkstr_regressed_Response_pT7'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                    
                        hist_dict['H_GenPi_Eta_Lrg_simtrkstrResponse_pT7'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Lrg_simtrkstr_regressed_Response_pT5'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Lrg_ticlcand_Response_pT7'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_lyrClusterSum_Response_pT7'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Lrg_Response_pT7'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_tracking_pT7'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_Lorentz_Response_pT7'].Fill(Lsum.Pt()/igen.pt())

            elif igen.pt() > 14.0 and igen.pt() <16.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Lrg_trkstrResponse_pT8'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_trkstr_regressed_Response_pT8'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                    
                        hist_dict['H_GenPi_Eta_Lrg_simtrkstrResponse_pT8'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Lrg_simtrkstr_regressed_Response_pT5'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Lrg_ticlcand_Response_pT8'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_lyrClusterSum_Response_pT8'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Lrg_Response_pT8'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_tracking_pT8'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_Lorentz_Response_pT8'].Fill(Lsum.Pt()/igen.pt())

            elif igen.pt() > 16.0 and igen.pt() <18.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Lrg_trkstrResponse_pT9'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_trkstr_regressed_Response_pT9'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                    
                        hist_dict['H_GenPi_Eta_Lrg_simtrkstrResponse_pT9'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Lrg_simtrkstr_regressed_Response_pT5'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Lrg_ticlcand_Response_pT9'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_lyrClusterSum_Response_pT9'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Lrg_Response_pT9'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_tracking_pT9'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_Lorentz_Response_pT9'].Fill(Lsum.Pt()/igen.pt())

            elif igen.pt() > 18.0 and igen.pt() <20.0:
                if kticlMindR >= 0:    
                    #hist_dict['H_GenPi_Eta_Lrg_trkstrResponse_pT10'].Fill(trkstr_p4.Pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_trkstr_regressed_Response_pT10'].Fill(trkstr_p4_regressed.Pt()/igen.pt())
                    if ksimtrkstrMindR>=0:                    
                        hist_dict['H_GenPi_Eta_Lrg_simtrkstrResponse_pT10'].Fill(ksimtrkstr.raw_energy()/igen.energy())
                    #hist_dict['H_GenPi_Eta_Lrg_simtrkstr_regressed_Response_pT5'].Fill(simtrkstr_p4_regressed.Pt()/igen.pt())
                    hist_dict['H_GenPi_Eta_Lrg_ticlcand_Response_pT10'].Fill(kticl.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_lyrClusterSum_Response_pT10'].Fill(EsumCls/igen.energy())
                #if jpfMindR >= 0:
                    #hist_dict['H_GenPi_Eta_Lrg_Response_pT10'].Fill(jpf.pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_tracking_pT10'].Fill(jpf.trackRef().pt()/igen.pt())
                    #hist_dict['H_GenPi_Eta_Lrg_Lorentz_Response_pT10'].Fill(Lsum.Pt()/igen.pt())

    #print 'npfcands :', npf
    #print "match: ",igen.pt(),sum_pt

    '''
    #photons
    npf=0
    for i,j in enumerate(gedPhotons.product()):
        npf=npf+1
        #   print "gedPhotons: pt %17.13f eta %18.14f pdgId %5d " % ( j.pt(), j.eta(), j.pdgId())
        #   print "nphoton: ",npf

    #electrons
    npf=0
    for i,j in enumerate(gedGsfElectrons.product()):
        npf=npf+1
        #       print "gedGsfElectrons: pt %17.13f eta %18.14f pdgId %5d " % ( j.pt(), j.eta(), j.pdgId())
        #       print "nelectron: ",npf

    # taus
        
    # jets

    # met
    '''

#Write HistOAograms to File

f = ROOT.TFile.Open("myfile04.root","RECREATE")

H_GenPi_Pt.Write()
H_GenPi_Eta.Write()
H_GenPi_Resp.Write()

for pre in ["energy","tracking","Response", "simtrkstrResponse", "simtrkstr_regressed_Response", "Tracking_Response", "Lorentz_Response", "trkstrResponse", "trkstr_regressed_Response", "ticlcand_Response", "lyrClusterSum_Response"]:
    for etaRange in ["Sml","Med","Lrg"]:
        for i in range(1, 11):
            hist_dict["H_GenPi_Eta_"+etaRange+"_"+pre+"_pT"+str(i)].Write()

#ROOT.TFile.Close(f)
f.Write()
f.Close()
