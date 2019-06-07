#pragma once

// Standard C includes
#include <cmath>

//------------------------------------------------------------------------
// RunningRegression
//------------------------------------------------------------------------
class RunningRegression
{
public:
    RunningRegression(double initialA = 1.0, double initialB = 0.0)
    : m_InitialA(initialA), m_InitialB(initialB), m_MeanX(0.0), m_MeanY(0.0), 
        m_VarX(0.0), m_CovXY(0.0), m_N(0)
    {
    }
    
    //------------------------------------------------------------------------
    // Public API
    //------------------------------------------------------------------------
    void update(double x, double y)
    {
        m_N++;
        const double n = (double)m_N;
        const double dx = x - m_MeanX;
        const double dy = y - m_MeanY;
        
        m_VarX += (((n - 1.0) / n) * dx * dx - m_VarX) / n;
        m_CovXY += (((n - 1.0) / n) * dx * dy - m_CovXY) / n;
        m_MeanX += dx / n;
        m_MeanY += dy / n;
    }
    
    double getA() const
    {
        return (m_N > 1) ? (m_CovXY / m_VarX) : m_InitialA;
    }
    
    double getB() const
    {
        return (m_N > 1) ? (m_MeanY - (getA() * m_MeanX)) : m_InitialB;
    }
    
    double estimate(double x) const
    {
        return (getA() * x) + getB();
    }
    
private:
    //------------------------------------------------------------------------
    // Members
    //------------------------------------------------------------------------
    const double m_InitialA;
    const double m_InitialB;
    
    double m_MeanX;
    double m_MeanY;
    
    double m_VarX;
    double m_CovXY;
    
    size_t m_N;
};