```vbscript
Sub CalculateStats()
    Dim wbA As Workbook
    Dim wbB As Workbook
    Dim wsA As Worksheet
    Dim wsB As Worksheet
    
    Dim rngDataPrecision As Range
    Dim rngDataRecall As Range
    Dim rngDataF1 As Range
    Dim rngDataAP As Range
    
    Dim meanValuePrecision As Double
    Dim stdDevValuePrecision As Double
    Dim meanValueRecall As Double
    Dim stdDevValueRecall As Double
    Dim meanValueF1 As Double
    Dim stdDevValueF1 As Double
    Dim meanValueAP As Double
    Dim stdDevValueAP As Double
    
    Dim rngMeanValue As Range
    Dim rngStdDevValue As Range
    
    Dim strFileName As String
    
    Dim startPos As String
    Dim modelName(1 To 5) As String
    
    modelName(1) = "LR"
    modelName(2) = "RF"
    modelName(3) = "XGB"
    modelName(4) = "SVM"
    modelName(5) = "TransformerClassifier"
    
    For k = 1 To 5 Step 1
    
        For i = 1 To 6 Step 1
        
            If i <= 5 Then
                strFileName = modelName(k) & "_n" & (i * 10)
            ElseIf k = 5 Then
                Exit For
            Else
                strFileName = modelName(k)
            End If
            
            ' 打开 A 文件
            Set wbA = Workbooks.Open("E:.\" & strFileName & ".csv") ' 更改为 A 文件的路径
            Set wsA = wbA.Sheets(strFileName) ' 更改为 A 文件中包含数据的工作表
            
            ' 打开 B 文件
            Set wbB = ThisWorkbook ' 假设该代码所在的文件就是 B 文件
            Set wsB = wbB.Sheets("Sheet1") ' 更改为 B 文件中要放置结果的工作表
            
            ' 获取数据范围
            With wsA
                Set rngDataPrecision = .Range("B2:B11")
                Set rngDataRecall = .Range("C2:C11")
                Set rngDataF1 = .Range("D2:D11")
                Set rngDataAP = .Range("E2:E11")
            End With
            
            ' 计算均值和样本标准差
            meanValuePrecision = Application.WorksheetFunction.Average(rngDataPrecision)
            stdDevValuePrecision = Application.WorksheetFunction.StDev(rngDataPrecision)
            
            meanValueRecall = Application.WorksheetFunction.Average(rngDataRecall)
            stdDevValueRecall = Application.WorksheetFunction.StDev(rngDataRecall)
            
            meanValueF1 = Application.WorksheetFunction.Average(rngDataF1)
            stdDevValueF1 = Application.WorksheetFunction.StDev(rngDataF1)
            
            meanValueAP = Application.WorksheetFunction.Average(rngDataAP)
            stdDevValueAP = Application.WorksheetFunction.StDev(rngDataAP)
            
            ' 将结果写入 B 文件
            With wsB
                Set rngMeanValue = .Range("B2").Offset(i - 1, (k - 1) * 2)
                Set rngStdDevValue = rngMeanValue.Offset(0, 1)
                
                rngMeanValue.Value = meanValuePrecision
                rngStdDevValue.Value = stdDevValuePrecision
                rngMeanValue.Offset(7, 0).Value = meanValueRecall
                rngStdDevValue.Offset(7, 0).Value = stdDevValueRecall
                rngMeanValue.Offset(14, 0).Value = meanValueF1
                rngStdDevValue.Offset(14, 0).Value = stdDevValueF1
                rngMeanValue.Offset(21, 0).Value = meanValueAP
                rngStdDevValue.Offset(21, 0).Value = stdDevValueAP
            End With
            ' 关闭 A 文件
            wbA.Close SaveChanges:=False
        Next i
    Next k
End Sub
```

