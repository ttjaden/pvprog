from scipy.io import loadmat
import numpy as np
import math

class BatProg:
    def __init__(self, dt: int = 60, P_stc: int = 5, C_bu: int = 5,
                 P_inv: float = 2.5, p_gfl: float = 0.5, eta_batt: float = 0.95,
                 eta_inv: float = 0.94, tf_past: int = 3, tf_prog: int = 15): 
        """
        Some hard coded values.

        Parameters
        ----------
        dt : numeric
            Time increment in seconds
        P_stc : numeric
            nominal power of the PV generator under STC test conditions in kWp
        C_bu : numeric
            usable battery capacity in kWh
        P_inv : numeric
            Nominal power of the battery inverter in kW
        p_gfl : numeric (0...1)
            specific grid feed-in limit in kW/kWp
        eta_batt : numeric (0...1)
            Efficiency of battery storage (without AC/DC conversion)
        eta_inv : numeric (0...1)
            Efficiency of the battery inverter
        tf_past : numeric
            Look-back time window in h
        tf_prog : numeric
            Forecast horizon in h
        """
        self.dt=dt
        self.P_stc=P_stc
        self.C_bu=C_bu
        self.P_inv=P_inv
        self.p_gfl=p_gfl
        self.eta_batt=eta_batt
        self.eta_inv=eta_inv
        self.tf_past=tf_past
        self.tf_prog=tf_prog
    
    def batt_sim(self, P_b: float, soc_0: float):
        """ 
        Simple battery storage model in which conversion losses 
        are accounted for by constant loss factors.
        Source: J. Weniger: Dimensionierung und Netzintegration von
        PV-Speichersystemen. Masterarbeit, Hochschule für Technik und Wirtschaft
        HTW Berlin, 2013

        Parameters
        ----------
        P_b : numeric
            requestet Power of battery in this timestep (positiv: charge, negativ: discharge)
        soc_0 : numeric (0...1)
            Battery state of charge before this timestep

        Returns
        -------
        P_b : numeric
            actual power of battery in this timestep (positiv: charge, negativ: discharge)
        soc : numeric  (0...1)
            Battery state of charge after this timestep
    
         Mögliche AC-seitige Batterieleistung auf die
         Batteriewechselrichter-Nennleistung begrenzen
        """
        P_b=np.maximum(-self.P_inv*1000,np.minimum(self.P_inv*1000,P_b))
        #Batteriespeicherinhalt im Zeitschritt zuvor
        E_b0=soc_0*self.C_bu*1000
        if P_b>=0:# %Batterieladung
            # Mögliche DC-seitige Batterieleistung unter Berücksichtigung des
            # Batteriewechselrichter-Wirkungsgrads bestimmen
            P_b=P_b*self.eta_inv
            #Ladung
            E_b=np.minimum(self.C_bu*1000, E_b0+self.eta_batt*P_b*self.dt/3600)
            # Anpassung der wirklich genutzten Leistung
            P_b=np.minimum(P_b,(self.C_bu*1000-E_b0)/(self.eta_batt*self.dt/3600))
        
        else:# % Batterieentladung
            #Mögliche DC-seitige Batterieleistung unter Berücksichtigung des
            # Batteriewechselrichter-Wirkungsgrads bestimmen
            P_b=P_b/self.eta_inv
            #Entladung
            E_b=np.maximum(0, E_b0+P_b*self.dt/3600)
            #Anpassung der wirklich genutzten Leistung
            P_b=np.maximum(P_b,(-E_b0)/(self.dt/3600))

        #Realisierte AC-seitige Batterieleistung
        if P_b >0:#Ladung
            P_b=P_b/self.eta_inv
        else:#Entladung
            P_b=P_b*self.eta_inv
        #Ladezustand
        soc=E_b/(self.C_bu*1000)
        return(P_b,soc)

    def simu_erg(self,P_pv,P_ld,P_b):
        """
        Calculation of the relative power flows and annual energy balances. 
        The autarky level and the regulation losses are also determined.

        Parameters
        ----------
        P_pv : array
            PV power output in W
        P_ld : array
            household electrical load (load demand) in W
        P_b : array
            Battery power in W (positive: charge, negative: discharge)

        Returns
        -------
        a : numeric (0...1)
            degree of self-sufficiency
        v : numeric  (0...1)
            regulation losses 
        pf : dictionary
            with the following colums
            P_pv: PV power output in W
            P_ld: household electrical load (load demand) in W
            P_d: Differential power (PV power minus household load) in W
            P_du: Directly consumed PV power (direct usage) in W
            P_bc: Battery charge power in W
            P_bd: Battery discharge power in W
            P_gf: grid feed-in power in W
            P_gs: grid supply power in W
            P_ct: Power of regulation losses in W
        eb : dictionary
            with the following colums
            E_pv: generated PV energy in MWh/a
            E_ld: household electricity demand (load demand) in MWh/a
            E_du: PV energy directly consumed (direct usage) in MWh/a
            E_bc: battery charge in MWh/a
            E_bd: battery discharge in MWh/a
            E_gf: grid feed-in in MWh/a
            E_gs: grid supply in MWh/a
            E_ct: PV energy curtailment in MWh/a
        """
        pf=dict()
        pf['P_pv']=P_pv
        pf['P_ld']=P_ld
        pf['P_d']=P_pv-P_ld
        pf['P_du']=np.minimum(P_pv,P_ld)
        pf['P_bc']=np.maximum(0,P_b)
        pf['P_bd']=abs(np.minimum(0,P_b))
        pf['P_gf']=np.maximum(0,np.minimum(self.P_stc*1000*self.p_gfl,pf['P_d']-pf['P_bc']))
        pf['P_gs']=abs(np.minimum(0,pf['P_d']+pf['P_bd']))
        pf['P_ct']=P_pv-pf['P_du']-pf['P_bc']-pf['P_gf']

        eb=dict()
        eb['E_pv']=P_pv.mean()*8.76
        eb['E_ld']=P_ld.mean()*8.76
        eb['E_du']=pf['P_du'].mean()*8.76
        eb['E_bc']=pf['P_bc'].mean()*8.76
        eb['E_bd']=pf['P_bd'].mean()*8.76
        eb['E_gf']=pf['P_gf'].mean()*8.76
        eb['E_gs']=pf['P_gs'].mean()*8.76
        eb['E_ct']=pf['P_ct'].mean()*8.76

        a = (eb['E_du']+eb['E_bd'])/eb['E_ld']
        v = eb['E_ct']/eb['E_pv']
        return (a,v,pf,eb)

    def prog4pv(self,time,p_pv):
        """Generation of PV forecasts based on the historical measured values of 
        PV power depending on the forecast horizon and look-back time window.

        Source: J. Bergner, J. Weniger, T. Tjaden, V. Quaschning: Verbesserte
        Netzintegration von PV-Speichersystemen durch Einbindung lokal
        erstellter PV- und Lastprognosen. 30. Symposium Photovoltaische
        Solarenergie. Bad Staffelstein, 2015

        Parameters
        ----------
        time : array
            timestamps in datenum format
        p_pv : array
            PV power output in W
        
        Returns
        -------
        p_pvf : array
            predicted PV power output in W
        """
        # Vorinitialisierung
        p_pvmax=np.zeros(len(time))
        KTF=np.zeros(len(time))
        p_pvf=np.zeros((np.int64(len(time)*self.dt/900),math.ceil(self.tf_prog*4)))
        # Tagesverlauf der maximalen PV-Leistungsabgabe aus den Messwerten der
        # vergangenen 10 Tage bestimmen
        for t in range(int(86400/self.dt)-1, len(time)-int(86400/self.dt), int(86400/self.dt)):
            #Anzahl der Tage, die zurückgeguckt wird (max. 10 Tage)
            d_pv=int(np.minimum(math.ceil(t*self.dt/86400),10))
            #% spezifische PV-Leistung während des Zeitraums  
            p_pvsel=p_pv[t-d_pv*int(86400/self.dt)+1:t+1]
            #maximalen Tagesverlauf der PV-Leistung bestimme
            p_pvmax[t:t+int(86400/self.dt)]=(np.max(np.reshape(p_pvsel,(int(86400/self.dt),d_pv),order='F'),1))

        #Nachtindikator (Zeitraum ohne PV-Erzeugung) bestimmen
        n=p_pv<=0
        #PV-Leistung und max. PV-Leistung für Zeitraum mit PV-Erzeugung
        p_pv_day=p_pv[~n]
        pv_max_day=p_pvmax[~n]
        E_pv_past=np.zeros(sum(~n))
        E_max=np.zeros(sum(~n))
        #Aktuelle und maximale PV-Energie im Rückblick-Zeitfenster bestimmen
        for t in range(self.tf_past*int(3600/self.dt),len(p_pv_day)):
            E_pv_past[t]=sum(p_pv_day[t-self.tf_past*int(3600/self.dt):t])
            E_max[t]=sum(pv_max_day[t-self.tf_past*int(3600/self.dt):t])
        # Verhältnis von aktueller zu maximaler PV-Energie (Wetterlage-Index KTF) im Rückblickzeitfenster berechnen
        k_TF=np.divide(E_pv_past,E_max)
        KTF[~n]=k_TF
        #15-Minutenmittelwerte von KTF und der maximalen PV-Leistungsabgabe p_pvmax
        KTF15=np.mean(np.reshape(KTF,(int(3600/self.dt/4),int(len(time)/(3600/self.dt/4))),order='F'),axis=0)
        p_pvmax15=np.mean(np.reshape(p_pvmax,(int(3600/self.dt/4),int(len(time)/(3600/self.dt/4))),order='F'),axis=0)
        #Zeitreihe p_pvmax15 zweimal verketten, um zum Ende der Jahressimulation auf die Maximalwerte des Jahresanfangs zurückzugreifen
        p_pvmax15=(np.append(p_pvmax15,p_pvmax15))
        #Messwertbasierte PV-Prognose erstellen: Multiplikation des aktuellen KTF15-Wertes mit dem Verlauf der maximalen PV-Leistung des Prognosehorizonts
        for t in range(0,len(p_pvf)):
            p_pvf[t,:]=np.maximum(0,np.minimum(1,KTF15[t]*(p_pvmax15[t:t+self.tf_prog*4])))
        # PV-Prognosen ohne Zahlenwert null setzen
        p_pvf[np.where(np.isnan(p_pvf))]=0
        return(p_pvf)
    
    def prog4ld(self,time,P_ld):
        """
        Generation of load forecasts based on the historical measured values of 
        the load. The average value of the past 15 min (current persistence) 
        and the load profile of the previous day (daily persistence) are weighted 
        differently over the forecast horizon.

        Source: J. Bergner, J. Weniger, T. Tjaden, V. Quaschning: Verbesserte
        Netzintegration von PV-Speichersystemen durch Einbindung lokal
        erstellter PV- und Lastprognosen. 30. Symposium Photovoltaische
        Solarenergie. Bad Staffelstein, 2015
        
        Parameters
        ----------
        time : array
            timestamps in datenum format
        P_ld : array
            household electrical load (load demand) in W
        
        Returns
        -------
        P_ldf : array
            predicted houshold electrical load in W
        time_f: array 
            array like time with value every 15 min.
        """
        #% Vorinitialisierung
        P_ldf=np.zeros((int(len(time)*self.dt/900),(self.tf_prog*4)))

        # 15 min-Zeitstempel für die Prognosen
        time_f=time[range(0,len(time)-int(900/self.dt)+1,int(900/self.dt))]
        # Lastprofil in 15-minütiger Auflösung ermitteln
        P_ld15=np.mean(np.reshape(P_ld,(int(900/self.dt),int(len(time)/int(900/self.dt))),order='F'),axis=0)
        #Gewichtungsfaktoren für die aktuelle Persistenz und Tagespersistenz über den Prognosehorizont variieren
        g1=1/math.exp(-0.1)*np.exp(-0.1*(np.arange(self.tf_prog*4)+1))#aktuelle Persistenz
        g2=1-g1#Tagespersistenz
        #Messwertbasierte Lastprognose erstellen: Variable Gewichtung von aktueller Persistenz und Tagespersistenz über den Prognosehorizont
        for t in range(96,len(time_f)):
            P_ldf[t,:]=g1*np.full(int(self.tf_prog*4),P_ld15[t-1])+g2*P_ld15[t-96:t-96+int(self.tf_prog*4)]
        
        return (P_ldf,time_f)

    def batt_prog(self,t,P_df,soc):
        """ 
        Creation of a schedule for the battery power over the forecast horizon of the 
        PV and load forecast. For this purpose, the virtual feed-in limit for the 
        period under consideration is minimized to such an extent that the excess PV 
        energy above this limit charges the battery storage as completely as possible. 
        
        Source: J. Weniger, V. Quaschning: Begrenzung der Einspeiseleistung von
        netzgekoppelten Photovoltaiksystemen mit Batteriespeichern. In: 28.
        Symposium Photovoltaische Solarenergie. Bad Staffelstein, 2013

        Further information on forecast-based battery charge planning:
        J. Bergner: Untersuchungen zu prognosebasierten Betriebsstrategien für
        PV-Speichersysteme. Berlin, Hochschule für Technik und Wirtschaft
        Berlin, Bachelorthesis, 2014 
        
        Parameters
        ----------
        t : numeric
            Time step
        P_df : array
            forecast of differential power in W (P_pvf-P_ldf)
        soc : array
            Battery state of charge

        Returns
        -------
        P_bf : array
            forcast power of battery for the next timesteps(positiv: charge, negativ: discharge)
        P_dfsel : array
            forcast of differential Power at this timestep
        
        """
        #aktueller Prognosezeitschritt
        t_fsel=math.floor(t*self.dt/900)
        # aktuelle Differenzleistungsprognose auswählen
        P_dfsel=P_df[t_fsel,:]
        # Batterieladezustand und Batterieinhalt im Zeitschritt zuvor
        soc_0=soc[t-1]
        E_b0=soc_0*self.C_bu*1000
        # Vorbereitung der Bestimmung der aktuellen virtuellen Einspeisegrenze durch Variation der virtuellen Einspeisegrenze in 0,01 kW/kWp-Schritten
        p_gflvir=np.reshape(np.repeat(np.arange(0,self.p_gfl+0.01,0.01),len(P_dfsel),0),[int(self.tf_prog*4),int(self.p_gfl*100+1)],order='F')
        # Prognostizierte überschüssige PV-Leistung
        P_sf=np.reshape(np.repeat(np.maximum(0,P_dfsel),p_gflvir.shape[1],0),p_gflvir.shape)
        # Idendifikation der minimalen virtuellen Einspeisegrenze, die über den Prognosehorizont eingehalten werden soll: Dabei soll die Energiemenge 
        #oberhalb dieser Grenze ausreichend sein, um den Batteriespeicher über den Prognosehorizont möglichst vollständig zu laden.
        value=(abs(np.sum(np.maximum(0,(P_sf-p_gflvir*self.P_stc*1000))*self.eta_batt*self.eta_inv*self.dt*900/self.dt/3600,axis=0)-(self.C_bu*1000-E_b0)))
        idx=np.where(value==np.min(value))[0][0]
        p_gflvir=p_gflvir[0,idx]
        # Batterieladeleistung über Prognosehorizont aus virtueller Einspeisegrenze ableiten
        P_bcf=np.maximum(0,P_dfsel-p_gflvir*self.P_stc*1000)
        # Batterieleistung aus Batterieladeleistung und Differenzleistung über Prognosehorizont bestimmen
        P_bf=np.round(np.minimum(P_bcf,P_dfsel))
        return (P_bf,P_dfsel)
        
    def err_ctrl(self,t,P_d,P_dfsel,P_bf):
        """
        Adjustment of the planned battery power to compensate for forecast errors. 
        For this purpose, the forecast charging power is corrected by a control system
        by the difference between the forecast and measured values.

         Source: J. Weniger, J. Bergner, V. Quaschning: Integration of PV power
         and load forecasts into the operation of residential PV battery systems.
         In: 4th Solar Integration Workshop. Berlin, 2014
         
        Parameters
        ----------
        t : numeric
            Time step
        P_d : array
            differential power in W (P_pv-P_ld)
        P_dfsel : array
            forcast of differential Power at this timestep 
        P_bf : array
            forcast power of battery for the next timesteps(positiv: charge, negativ: discharge)

        Returns
        -------
        P_b : numeric
            Battery power in this timestep (positiv: charge, negativ: discharge)
         """

        if P_d[t]>0:#(Leistungsüberschuss)
            """ % Anpassung der Ladeleistung, wenn die aktuelle Differenzleistung größer
                % null und überschüssige PV-Leistung vorhanden ist
                %
                % Batterieladeleistung wird angepasst, wenn eine der folgenden
                % Bedingungen erfüllt wird:
                %
                % (1) Die für den aktuellen Zeitschritt prognostizierte
                % Batterieleistung ist ungleich null
                % (2) Die aktuelle Differenzleistung ist größer als die max.
                % prognostizierte Einspeiseleistung (virtuelle Einspeisegrenze)
                % während des Prognosehorizonts
                % (3) Die aktuelle Differenzleistung übersteigt die max. zulässige
                % Einspeisegrenze"""
            
            if (P_bf[0]!=0) or (P_d[t]>np.max(P_dfsel-P_bf)) or (P_d[t]>self.p_gfl*self.P_stc*1000):
                """ % Aktuelle Ladeleistung um die Differenz zwischen der aktuellen
                    % Differenzleistung P_d(t) und der prognostizierten Differenzleistung
                    % P_dfsel(1) korrigieren. Dadurch wird gewährleistet, dass die
                    % zuvor ermittelte virtuelle Einspeisegrenze eingehalten wird"""
                    
                # Ladeleistung auf die Nennleistung des Batteriewechselrichters begrenzen
                P_b=np.maximum(0,P_bf[0]+P_d[t]-P_dfsel[0])
                # Ladeleistung auf die Nennleistung des Batteriewechselrichters begrenzen
                P_b=np.minimum(self.P_inv*1000,P_b)
            else:
                """ % Wenn keine der zuvor aufgeführten Bedingungen erfüllt wird, soll
                    % die aktuelle Batterieladeleistung auf null gesetzt werden.
                    % Dadurch wird eine stufige Anpassung der Einspeiseleistung
                    % verhindert."""
                P_b=0
        else:#% P_d(t)<0 (Leistungsdefizit)
            """ % Entladeleistung gemäß Leistungsdefizit anpassen und auf die Nennleistung des
                % Batteriewechselrichters begrenzen."""
            P_b=np.maximum(-self.P_inv*1000,P_d[t])  
        return(P_b)
      