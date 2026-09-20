"""PAGE-XML as Kraken and Transkribus write it."""

from htrbench.layout.pagexml import read_page_from_string, scale_factor

XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<PcGts xmlns="http://schema.primaresearch.org/PAGE/gts/pagecontent/2013-07-15">
  <Metadata><Creator>prov=READ-COOP:name=TrHtr:model_id=265149:date=06_09_2026</Creator></Metadata>
  <Page imageFilename="0097_p1.jpg" imageWidth="1000" imageHeight="2000">
    <ReadingOrder><OrderedGroup id="ro">
      <RegionRefIndexed index="0" regionRef="main"/><RegionRefIndexed index="1" regionRef="margin"/>
    </OrderedGroup></ReadingOrder>
    <TextRegion id="margin"><Coords points="0,0 10,0 10,10"/>
      <TextLine id="m1"><Coords points="0,0 10,0 10,10 0,10"/><TextEquiv><Unicode>Concl.</Unicode></TextEquiv></TextLine>
    </TextRegion>
    <TextRegion id="main"><Coords points="0,0 10,0 10,10"/>
      <TextLine id="l1"><Coords points="20,20 90,20 90,40 20,40"/>
        <TextEquiv index="1"><Unicode>second reading</Unicode></TextEquiv><TextEquiv index="0"><Unicode>erste Zeile</Unicode></TextEquiv>
      </TextLine>
      <TextLine id="l2"><Coords points="20,50 90,50 90,70 20,70"/></TextLine>
    </TextRegion>
  </Page>
</PcGts>"""


def test_lines_follow_the_declared_reading_order():
    page = read_page_from_string(XML)
    assert [ln.id for ln in page.lines()] == ["l1", "l2", "m1"]
    assert page.text() == "erste Zeile\n\nConcl."


def test_metadata_the_transkribus_importer_relies_on():
    page = read_page_from_string(XML)
    assert "model_id=265149" in page.creator and page.image_filename == "0097_p1.jpg"
    assert scale_factor(page, 500, 1000) == (0.5, 0.5)
    assert page.lines()[0].polygon == [(20, 20), (90, 20), (90, 40), (20, 40)]
