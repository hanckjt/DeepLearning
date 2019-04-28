# -*- coding: utf-8 -*-

###########################################################################
## Python code generated with wxFormBuilder (version Oct 26 2018)
## http://www.wxformbuilder.org/
##
## PLEASE DO *NOT* EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc


###########################################################################
## Class MnistTestUI
###########################################################################

class MnistTestUI(wx.Frame):

    def __init__(self, parent=None):
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title=u"MnistTest", pos=wx.DefaultPosition,
                          size=wx.Size(500, 400), style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)

        self.SetSizeHints(wx.DefaultSize, wx.DefaultSize)
        self.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_3DLIGHT))

        self.m_statusBar1 = self.CreateStatusBar(1, wx.STB_SIZEGRIP, wx.ID_ANY)
        self.m_statusBar1.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_3DLIGHT))

        self.m_toolBar1 = self.CreateToolBar(wx.TB_HORIZONTAL, wx.ID_ANY)
        self.m_toolBar1.SetBackgroundColour(wx.SystemSettings.GetColour(wx.SYS_COLOUR_3DLIGHT))

        self.m_button1 = wx.Button(self.m_toolBar1, wx.ID_ANY, u"Clear", wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_toolBar1.AddControl(self.m_button1)
        self.m_button2 = wx.Button(self.m_toolBar1, wx.ID_ANY, u"Predict", wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_toolBar1.AddControl(self.m_button2)
        self.m_toolBar1.Realize()

        bSizer1 = wx.BoxSizer(wx.HORIZONTAL)

        sbSizer1 = wx.StaticBoxSizer(wx.StaticBox(self, wx.ID_ANY, u"Draw"), wx.VERTICAL)

        self.m_drawPanel = wx.Panel(sbSizer1.GetStaticBox(), wx.ID_ANY, wx.Point(-1, -1), wx.Size(280, 280),
                                    wx.TAB_TRAVERSAL)
        self.m_drawPanel.SetForegroundColour(wx.Colour(0, 0, 0))
        self.m_drawPanel.SetBackgroundColour(wx.Colour(255, 255, 255))
        self.m_drawPanel.SetMaxSize(wx.Size(280, 280))

        sbSizer1.Add(self.m_drawPanel, 1, wx.EXPAND | wx.ALL, 5)

        bSizer1.Add(sbSizer1, 1, wx.ALIGN_CENTER | wx.ALIGN_CENTER_HORIZONTAL | wx.ALIGN_CENTER_VERTICAL | wx.EXPAND, 5)

        sbSizer2 = wx.StaticBoxSizer(wx.StaticBox(self, wx.ID_ANY, u"Result"), wx.VERTICAL)

        bSizer23 = wx.BoxSizer(wx.HORIZONTAL)

        self.m_staticText13 = wx.StaticText(sbSizer2.GetStaticBox(), wx.ID_ANY, u"0", wx.DefaultPosition,
                                            wx.DefaultSize, 0)
        self.m_staticText13.Wrap(-1)

        bSizer23.Add(self.m_staticText13, 0, wx.ALL, 5)

        self.m_gauge13 = wx.Gauge(sbSizer2.GetStaticBox(), wx.ID_ANY, 100, wx.DefaultPosition, wx.DefaultSize,
                                  wx.GA_HORIZONTAL)
        self.m_gauge13.SetValue(0)
        bSizer23.Add(self.m_gauge13, 0, wx.ALL, 5)

        sbSizer2.Add(bSizer23, 1, wx.EXPAND, 5)

        bSizer2 = wx.BoxSizer(wx.HORIZONTAL)

        self.m_staticText1 = wx.StaticText(sbSizer2.GetStaticBox(), wx.ID_ANY, u"1", wx.DefaultPosition, wx.DefaultSize,
                                           0)
        self.m_staticText1.Wrap(-1)

        bSizer2.Add(self.m_staticText1, 0, wx.ALL, 5)

        self.m_gauge1 = wx.Gauge(sbSizer2.GetStaticBox(), wx.ID_ANY, 100, wx.DefaultPosition, wx.DefaultSize,
                                 wx.GA_HORIZONTAL)
        self.m_gauge1.SetValue(0)
        bSizer2.Add(self.m_gauge1, 0, wx.ALL, 5)

        sbSizer2.Add(bSizer2, 1, wx.EXPAND, 5)

        bSizer21 = wx.BoxSizer(wx.HORIZONTAL)

        self.m_staticText11 = wx.StaticText(sbSizer2.GetStaticBox(), wx.ID_ANY, u"2", wx.DefaultPosition,
                                            wx.DefaultSize, 0)
        self.m_staticText11.Wrap(-1)

        bSizer21.Add(self.m_staticText11, 0, wx.ALL, 5)

        self.m_gauge11 = wx.Gauge(sbSizer2.GetStaticBox(), wx.ID_ANY, 100, wx.DefaultPosition, wx.DefaultSize,
                                  wx.GA_HORIZONTAL)
        self.m_gauge11.SetValue(0)
        bSizer21.Add(self.m_gauge11, 0, wx.ALL, 5)

        sbSizer2.Add(bSizer21, 1, wx.EXPAND, 5)

        bSizer22 = wx.BoxSizer(wx.HORIZONTAL)

        self.m_staticText12 = wx.StaticText(sbSizer2.GetStaticBox(), wx.ID_ANY, u"3", wx.DefaultPosition,
                                            wx.DefaultSize, 0)
        self.m_staticText12.Wrap(-1)

        bSizer22.Add(self.m_staticText12, 0, wx.ALL, 5)

        self.m_gauge12 = wx.Gauge(sbSizer2.GetStaticBox(), wx.ID_ANY, 100, wx.DefaultPosition, wx.DefaultSize,
                                  wx.GA_HORIZONTAL)
        self.m_gauge12.SetValue(0)
        bSizer22.Add(self.m_gauge12, 0, wx.ALL, 5)

        sbSizer2.Add(bSizer22, 1, wx.EXPAND, 5)

        bSizer24 = wx.BoxSizer(wx.HORIZONTAL)

        self.m_staticText14 = wx.StaticText(sbSizer2.GetStaticBox(), wx.ID_ANY, u"4", wx.DefaultPosition,
                                            wx.DefaultSize, 0)
        self.m_staticText14.Wrap(-1)

        bSizer24.Add(self.m_staticText14, 0, wx.ALL, 5)

        self.m_gauge14 = wx.Gauge(sbSizer2.GetStaticBox(), wx.ID_ANY, 100, wx.DefaultPosition, wx.DefaultSize,
                                  wx.GA_HORIZONTAL)
        self.m_gauge14.SetValue(0)
        bSizer24.Add(self.m_gauge14, 0, wx.ALL, 5)

        sbSizer2.Add(bSizer24, 1, wx.EXPAND, 5)

        bSizer25 = wx.BoxSizer(wx.HORIZONTAL)

        self.m_staticText15 = wx.StaticText(sbSizer2.GetStaticBox(), wx.ID_ANY, u"5", wx.DefaultPosition,
                                            wx.DefaultSize, 0)
        self.m_staticText15.Wrap(-1)

        bSizer25.Add(self.m_staticText15, 0, wx.ALL, 5)

        self.m_gauge15 = wx.Gauge(sbSizer2.GetStaticBox(), wx.ID_ANY, 100, wx.DefaultPosition, wx.DefaultSize,
                                  wx.GA_HORIZONTAL)
        self.m_gauge15.SetValue(0)
        bSizer25.Add(self.m_gauge15, 0, wx.ALL, 5)

        sbSizer2.Add(bSizer25, 1, wx.EXPAND, 5)

        bSizer26 = wx.BoxSizer(wx.HORIZONTAL)

        self.m_staticText16 = wx.StaticText(sbSizer2.GetStaticBox(), wx.ID_ANY, u"6", wx.DefaultPosition,
                                            wx.DefaultSize, 0)
        self.m_staticText16.Wrap(-1)

        bSizer26.Add(self.m_staticText16, 0, wx.ALL, 5)

        self.m_gauge16 = wx.Gauge(sbSizer2.GetStaticBox(), wx.ID_ANY, 100, wx.DefaultPosition, wx.DefaultSize,
                                  wx.GA_HORIZONTAL)
        self.m_gauge16.SetValue(0)
        bSizer26.Add(self.m_gauge16, 0, wx.ALL, 5)

        sbSizer2.Add(bSizer26, 1, wx.EXPAND, 5)

        bSizer27 = wx.BoxSizer(wx.HORIZONTAL)

        self.m_staticText17 = wx.StaticText(sbSizer2.GetStaticBox(), wx.ID_ANY, u"7", wx.DefaultPosition,
                                            wx.DefaultSize, 0)
        self.m_staticText17.Wrap(-1)

        bSizer27.Add(self.m_staticText17, 0, wx.ALL, 5)

        self.m_gauge17 = wx.Gauge(sbSizer2.GetStaticBox(), wx.ID_ANY, 100, wx.DefaultPosition, wx.DefaultSize,
                                  wx.GA_HORIZONTAL)
        self.m_gauge17.SetValue(0)
        bSizer27.Add(self.m_gauge17, 0, wx.ALL, 5)

        sbSizer2.Add(bSizer27, 1, wx.EXPAND, 5)

        bSizer28 = wx.BoxSizer(wx.HORIZONTAL)

        self.m_staticText18 = wx.StaticText(sbSizer2.GetStaticBox(), wx.ID_ANY, u"8", wx.DefaultPosition,
                                            wx.DefaultSize, 0)
        self.m_staticText18.Wrap(-1)

        bSizer28.Add(self.m_staticText18, 0, wx.ALL, 5)

        self.m_gauge18 = wx.Gauge(sbSizer2.GetStaticBox(), wx.ID_ANY, 100, wx.DefaultPosition, wx.DefaultSize,
                                  wx.GA_HORIZONTAL)
        self.m_gauge18.SetValue(0)
        bSizer28.Add(self.m_gauge18, 0, wx.ALL, 5)

        sbSizer2.Add(bSizer28, 1, wx.EXPAND, 5)

        bSizer29 = wx.BoxSizer(wx.HORIZONTAL)

        self.m_staticText19 = wx.StaticText(sbSizer2.GetStaticBox(), wx.ID_ANY, u"9", wx.DefaultPosition,
                                            wx.DefaultSize, 0)
        self.m_staticText19.Wrap(-1)

        bSizer29.Add(self.m_staticText19, 0, wx.ALL, 5)

        self.m_gauge19 = wx.Gauge(sbSizer2.GetStaticBox(), wx.ID_ANY, 100, wx.DefaultPosition, wx.DefaultSize,
                                  wx.GA_HORIZONTAL)
        self.m_gauge19.SetValue(0)
        bSizer29.Add(self.m_gauge19, 0, wx.ALL, 5)

        sbSizer2.Add(bSizer29, 1, wx.EXPAND, 5)

        bSizer1.Add(sbSizer2, 1, wx.EXPAND, 5)

        self.SetSizer(bSizer1)
        self.Layout()

        self.Centre(wx.BOTH)

        self.Bind(wx.EVT_CLOSE, self.onClose)

        self.m_drawPanel.Bind(wx.EVT_PAINT, self.onPaint)
        self.m_drawPanel.Bind(wx.EVT_LEFT_DOWN, self.onLeftDown)
        self.m_drawPanel.Bind(wx.EVT_LEFT_UP, self.onLeftUp)
        self.m_drawPanel.Bind(wx.EVT_MOTION, self.onMotion)

        self.isDrawing = False
        self.mnistArray = [28][28]

    def onPaint(self, e):
        dc = wx.PaintDC(self.m_drawPanel)
        pen = wx.Pen(wx.Colour(0, 0, 0))
        brush = wx.Brush(wx.Colour(0, 0, 0))
        dc.SetPen(pen)
        self.drawGrid(dc)

    def drawGrid(self, dc):
        for p in range(0, 280, 10):
            dc.DrawLine(p, 0, p, 280)
            dc.DrawLine(0, p, 280, p)

    def onClose(self, e):
        quit()

    def onLeftDown(self, event):
        self.isDrawing = True

    def onLeftUp(self, event):
        self.isDrawing = False

    def onMotion(self, event):
        event.Skip()

    def __del__(self):
        pass


if __name__ == '__main__':
    app = wx.App()
    frame = MnistTestUI()
    frame.Show()
    app.MainLoop()
