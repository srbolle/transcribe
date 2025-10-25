# Meta-flytskjema: Felles kjerneflyt + grener (Vilde/ORT og Tove/GYN)

Dette er en norsk versjon ment for lesing i Markdown/VS Code med Mermaid-støtte.

```mermaid
flowchart TD

  %% ===================
  %% Felles kjerneflyt
  %% ===================
  subgraph F[Felles kjerneflyt]
    A[Henvisning mottas] --> B[Sekretær kontrollerer innhold]
    B --> C{Riktig seksjon/fag?}
    C -- Ja --> D[Vurdering av lege (poliklinikk)]
    C -- Nei --> C2[Rett seksjon/fag og ruter til rett arbeidsgruppe] --> D

    D --> E{Operasjon indikert?}
    E -- Nei --> F[Avslutt/oppfølging uten operasjon]
    E -- Ja --> G[Formell beslutning om operasjon]

    G --> H[Planlegger kobler henvisning og operasjon]
    H --> I[Kontroller metadata: omsorgsnivå, avdeling, lokasjon, diagnosegruppe, kontakt]
    I --> J{Direkte time eller tentativ måned?}

    J -- Direkte time --> K[Finn ledig slot/stue]
    K --> L[Sett operatør, utstyr, start/slutt]
    L --> M[Innkallingsbrev]
    M --> N[På operasjonsprogram]

    J -- Tentativ måned --> V1[Sett tentativ måned]
    V1 --> V2[Ventelistebrev]
    V2 --> V3[Hent fra venteliste ved kapasitet]
    V3 --> L

    %% Kontrollpunkter
    I --> Q{Uklart/uvanlig? Feil operatør/indikasjon?}
    Q -- Ja --> R[Avklar med lege (gulapp) før brev/plan]
    R --> I
    Q -- Nei --> J

    %% Ressurser
    L --> Z{Riktig utstyr/program/stue?}
    Z -- Ja --> M
    Z -- Nei --> Z2[Koordiner med lege / velg passende dag/stue] --> K
  end

  %% ===================
  %% Gren: ORT (Vilde)
  %% ===================
  subgraph O[Gren ORT (Vilde)]
    O1[Operatørbeslutning fylles ut av lege] --> O2[Operatørbeslutning til LHK-mappe]
    O2 --> O3[Arbeidsgruppe: Ortelektivoperasjon Tromsø]
    O3 --> H
  end

  %% ===================
  %% Gren: GYN (Tove)
  %% ===================
  subgraph GY[Gren GYN (Tove)]
    G0[KVIFØ-mottak: sekretærer tømmer mottakspostkasse på tvers av lokasjon] --> G1{Føde vs kvinnesykdom/kvinnekreft}
    G1 -- Føde --> G1A[Send til jordmor-arbeidsgruppe] --> D
    G1 -- Kvinnesykdom/kvinnekreft --> D

    D --> G2[Poliklinikknotat med operasjonsbeskjed]
    G2 --> G3[Sekretær oppretter sekundærhenvisning]
    G3 --> G4[Lege vurderer sekundærhenvisning
               og setter diagnosegruppe, prioritet (uker), omsorgsnivå]
    G4 --> H

    %% Kapasitets-/produksjonsplanlegging
    V3 --> G5[Gruppér like inngrep (f.eks. TVT-dager, 3 laparoskopier/dag)]
    G5 --> G6[Matche kirurg-kompetanse og blokk/ukeplan]
    G6 --> K
  end
```

Tips
- Åpne denne filen i en Markdown-viewer med Mermaid-støtte for å se diagrammet.
- Trenger du SVG/PNG-eksport, si ifra – jeg kan lage en egen ASCII-/CLI-vennlig versjon (uten æ/ø/å og spesialtegn) og eksportere for deg.
